#!/usr/bin/env python3
"""
NVIDIA CC Manager For Kubernetes (Python Implementation)

A Kubernetes component that enables required CC mode on supported NVIDIA GPUs
based on node labels. This is a Python reimplementation of the Go version,
utilizing NVIDIA's gpu-admin-tools for GPU management.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

GPU_ADMIN_TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gpu-admin-tools'))
sys.path.insert(0, str(GPU_ADMIN_TOOLS_PATH))

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

# Import gpu-admin-tools
try:
    from nvidia_gpu_tools import Gpu, NvSwitch
    from pci.devices import find_gpus, find_devices_from_string
    from gpu import GpuError
except ImportError as e:
    print(f"Error importing gpu-admin-tools: {e}", file=sys.stderr)
    print(f"GPU tools path: {GPU_ADMIN_TOOLS_PATH}", file=sys.stderr)
    sys.exit(1)

from gpu_operator_eviction import (
    fetch_current_component_labels,
    evict_gpu_operator_components,
    reschedule_gpu_operator_components,
    set_cc_mode_state_label
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('k8s-cc-manager')


# Constants
CC_MODE_CONFIG_LABEL = 'nvidia.com/cc.mode'
READINESS_FILE = os.environ.get('CC_READINESS_FILE', '/run/nvidia/validations/.cc-manager-ctr-ready')


def create_readiness_file():
    """
    Create the readiness file to indicate the CC manager container is ready.
    This is used by NVIDIA GPU Operator validation framework.
    """
    try:
        readiness_path = Path(READINESS_FILE)
        readiness_path.parent.mkdir(parents=True, exist_ok=True)
        readiness_path.touch()
        logger.info(f"Created readiness file: {READINESS_FILE}")
    except Exception as e:
        logger.warning(f"Failed to create readiness file {READINESS_FILE}: {e}")
        # Don't fail the application if readiness file can't be created


class CCManager:
    """Manages NVIDIA GPU Confidential Computing mode based on Kubernetes node labels."""
    
    def __init__(self, node_name: str, default_mode: str = ''):
        """
        Initialize the CC Manager.
        
        Args:
            node_name: Name of the Kubernetes node this manager runs on
            default_mode: Default CC mode to use if no label is set
        """
        self.operator_namespace = os.environ.get('OPERATOR_NAMESPACE', 'gpu-operator')
        self.evict_operator_components = os.environ.get(
            'EVICT_OPERATOR_COMPONENTS', 'true'
        ).lower() == 'true'
        self.node_name = node_name
        self.default_mode = default_mode
        self.current_label = None
        self.current_rv = None
        self.last_label = None
        self.max_consecutive_errors = 10
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig from default location")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes configuration: {e}")
                raise
        
        self.v1 = client.CoreV1Api()
        logger.info(f"Initialized CC Manager for node: {node_name}")
        logger.info(f"Default CC mode: {default_mode or '(none)'}")
    
    def get_cc_capable_gpus(self) -> list:
        """
        Discover CC-capable GPUs on the node.

        Returns:
            List of Gpu objects for CC-capable GPUs
        """
        cc_gpus = []
        all_devices, _ = find_gpus()
        for device in all_devices:
            # Skip NVSwitches - they don't support CC mode
            if device.is_nvswitch():
                continue

            # Verify CC is supported
            if not device.is_cc_query_supported:
                logger.warning(f"GPU {device.bdf} does not support CC mode query")
                continue

            cc_gpus.append(device)
            logger.info(f"Found CC-capable GPU: {device.bdf} - {device.name}")

        return cc_gpus

    def validate_mode(self, mode: str) -> bool:
        """
        Validate that the mode is supported.

        Args:
            mode: Mode string to validate

        Returns:
            True if mode is valid, False otherwise
        """
        valid_modes = ['on', 'off', 'devtools', 'ppcie']
        if mode not in valid_modes:
            logger.error(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
            return False
        return True

    def get_ppcie_capable_devices(self) -> tuple:
        """
        Discover PPCIe-capable devices (GPUs and NVSwitches) on the node.

        Returns:
            Tuple of (ppcie_gpus, ppcie_switches) lists
        """
        ppcie_gpus = []
        ppcie_switches = []

        # Find GPUs - filter out NVSwitches
        all_devices, _ = find_gpus()
        for device in all_devices:
            # Skip NVSwitches here, they're discovered separately below
            if device.is_nvswitch():
                continue

            if device.is_ppcie_query_supported:
                ppcie_gpus.append(device)
                logger.info(f"Found PPCIe-capable GPU: {device.bdf} - {device.name}")
            else:
                logger.debug(f"GPU {device.bdf} does not support PPCIe")

        # Find NVSwitches
        try:
            switches = find_devices_from_string("nvswitches")
            for switch in switches:
                if switch.is_ppcie_query_supported:
                    ppcie_switches.append(switch)
                    logger.info(f"Found PPCIe-capable NVSwitch: {switch.bdf} - {switch.name}")
                else:
                    logger.debug(f"NVSwitch {switch.bdf} does not support PPCIe")
        except Exception as e:
            logger.info(f"No NVSwitches found or error discovering them: {e}")

        return (ppcie_gpus, ppcie_switches)

    def _set_cc_mode_internal(self, mode: str) -> bool:
        """
        Internal method to set CC mode on all GPUs.

        Args:
            mode: Desired CC mode (e.g., 'on', 'off', 'devtools')

        Returns:
            True if successful, False otherwise
        """
        # Get all devices and filter to only GPUs (exclude NVSwitches)
        all_devices, _ = find_gpus()
        gpus = [d for d in all_devices if not d.is_nvswitch()]
        cc_gpus = self.get_cc_capable_gpus()

        # If the mode is not off and some of the devices are not cc-capable,
        # bail out here.
        if mode != 'off':
            if len(gpus) != len(cc_gpus):
                logger.error(f"Some GPUs are not cc-capable: {set(gpus) - set(cc_gpus)}")
                sys.exit(1)

        if not gpus:
            logger.warning("No GPUs to configure")
            return True

        if not mode:
            logger.info("No CC mode specified, skipping")
            return True

        if self.mode_is_set(gpus, mode):
            logger.info(f"All gpus already set to cc {mode}, skipping")
            set_cc_mode_state_label(self.v1, self.node_name, mode)
            return True

        if self.evict_operator_components:
            return self._set_cc_mode_with_eviction(gpus, mode)

        return self._set_cc_mode_direct(gpus, mode)
        
    def mode_is_set(self, gpus: list, mode: str) -> bool:
        """
        Checks if the CC mode is already set on all GPUs

        Args:
            gpus: List of Gpu objects
            mode: Desired CC mode (e.g., 'on', 'off', 'devtools')

        Returns:
            True if already set, False otherwise
        """
        for gpu in gpus:
            try:
                if gpu.query_cc_mode() != mode:
                    return False

            except Exception as e:
                logger.error(f"Unexpected error getting CC mode on {gpu.bdf}: {e}")
                return False
        return True

    def _set_cc_mode_direct(self, gpus: list, mode: str) -> bool:
        """
        Set CC mode on all specified GPUs.

        Args:
            gpus: List of Gpu objects
            mode: Desired CC mode (e.g., 'on', 'off', 'devtools')

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Setting CC mode to '{mode}' on {len(gpus)} GPU(s)")
        
        for gpu in gpus:
            try:
                # Check current mode
                current_mode = gpu.query_cc_mode()
                if current_mode == mode:
                    logger.info(f"GPU {gpu.bdf} already in CC mode '{mode}', skipping")
                    continue
                
                logger.info(f"Setting CC mode on GPU {gpu.bdf} from '{current_mode}' to '{mode}'")
                
                # Set CC mode
                gpu.set_cc_mode(mode)
                
                # Reset GPU to apply the mode (uses sysfs reset on Linux)
                logger.info(f"Resetting GPU {gpu.bdf} to apply CC mode")
                gpu.reset_with_os()
                
                # Wait for GPU to boot up after reset
                gpu.wait_for_boot()
                
                # Verify the mode was set
                new_mode = gpu.query_cc_mode()
                if new_mode != mode:
                    raise RuntimeError(
                        f"CC mode verification failed: expected '{mode}', got '{new_mode}'"
                    )
                
                logger.info(f"Successfully set CC mode to '{mode}' on GPU {gpu.bdf}")
                
            except GpuError as e:
                logger.error(f"GPU error setting CC mode on {gpu.bdf}: {e}")
                set_cc_mode_state_label(self.v1, self.node_name, 'failed')
                return False
            except Exception as e:
                logger.error(f"Unexpected error setting CC mode on {gpu.bdf}: {e}")
                set_cc_mode_state_label(self.v1, self.node_name, 'failed')
                return False
        
        logger.info(f"Successfully set CC mode to '{mode}' on all GPUs")
        set_cc_mode_state_label(self.v1, self.node_name, mode)
        return True
            
    def _set_cc_mode_with_eviction(self, gpus: list, mode: str) -> bool:
        """
        Evict GPU components, set CC mode on all specified GPUs,
        reschedule components.

        Args:
            gpus: List of Gpu objects
            mode: Desired CC mode (e.g., 'on', 'off', 'devtools')

        Returns:
            True if successful, False otherwise
        """
        component_labels = fetch_current_component_labels(self.v1, self.node_name)
        logger.info("Evicting GPU operator components before CC mode change")
        if not evict_gpu_operator_components(
            self.v1,
            self.node_name,
            self.operator_namespace,
            component_labels,
            timeout=300
        ):
            logger.error("Failed to evict GPU operator components")
            return False

        result = self._set_cc_mode_direct(gpus, mode)
        logger.info("Rescheduling GPU operator components")
        if not reschedule_gpu_operator_components(
            self.v1,
            self.node_name,
            component_labels
        ):
            logger.error("Failed to reschedule GPU operator components")
            result = False

        return result

    def ppcie_mode_is_set(self, devices: list, mode: str) -> bool:
        """
        Check if PPCIe mode is already set on all devices.

        Args:
            devices: List of Gpu/NvSwitch objects
            mode: Desired PPCIe mode ("on" or "off")

        Returns:
            True if already set, False otherwise
        """
        for device in devices:
            try:
                if device.query_ppcie_mode() != mode:
                    return False
            except Exception as e:
                logger.error(f"Error querying PPCIe mode on {device.bdf}: {e}")
                return False
        return True

    def _set_ppcie_mode_direct(self, devices: list, target_mode: str, label_mode: str) -> bool:
        """
        Set PPCIe mode on all devices (GPUs and NVSwitches).

        Args:
            devices: List of Gpu/NvSwitch objects
            target_mode: Desired PPCIe mode ("on" or "off")
            label_mode: Mode to set in label ("ppcie" or "off")

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Setting PPCIe mode to '{target_mode}' on {len(devices)} device(s)")

        for device in devices:
            try:
                device_type = "NVSwitch" if device.is_nvswitch() else "GPU"
                current_mode = device.query_ppcie_mode()

                if current_mode == target_mode:
                    logger.info(f"{device_type} {device.bdf} already in PPCIe '{target_mode}'")
                    continue

                logger.info(f"Setting PPCIe on {device_type} {device.bdf}: '{current_mode}' → '{target_mode}'")
                device.set_ppcie_mode(target_mode)

                logger.info(f"Resetting {device_type} {device.bdf}")
                device.reset_with_os()
                device.wait_for_boot()

                # Verify
                new_mode = device.query_ppcie_mode()
                if new_mode != target_mode:
                    raise RuntimeError(f"Verification failed: expected '{target_mode}', got '{new_mode}'")

                logger.info(f"Successfully set PPCIe '{target_mode}' on {device_type} {device.bdf}")

            except GpuError as e:
                logger.error(f"Device error on {device.bdf}: {e}")
                set_cc_mode_state_label(self.v1, self.node_name, 'failed')
                return False
            except Exception as e:
                logger.error(f"Unexpected error on {device.bdf}: {e}")
                set_cc_mode_state_label(self.v1, self.node_name, 'failed')
                return False

        set_cc_mode_state_label(self.v1, self.node_name, label_mode)
        return True

    def _set_ppcie_mode_with_eviction(self, devices: list, target_mode: str, label_mode: str) -> bool:
        """
        Evict GPU components, set PPCIe mode, reschedule components.

        Args:
            devices: List of Gpu/NvSwitch objects
            target_mode: Desired PPCIe mode ("on" or "off")
            label_mode: Mode to set in label ("ppcie" or "off")

        Returns:
            True if successful, False otherwise
        """
        component_labels = fetch_current_component_labels(self.v1, self.node_name)
        logger.info("Evicting GPU operator components for PPCIe mode change")

        if not evict_gpu_operator_components(
            self.v1,
            self.node_name,
            self.operator_namespace,
            component_labels,
            timeout=300
        ):
            logger.error("Failed to evict GPU operator components")
            return False

        result = self._set_ppcie_mode_direct(devices, target_mode, label_mode)

        logger.info("Rescheduling GPU operator components")
        if not reschedule_gpu_operator_components(
            self.v1,
            self.node_name,
            component_labels
        ):
            logger.error("Failed to reschedule GPU operator components")
            return False

        return result

    def set_ppcie_mode(self, mode: str) -> bool:
        """
        Orchestrate PPCIe mode setting on GPUs and NVSwitches.

        Args:
            mode: Desired mode ("ppcie" or "off")

        Returns:
            True if successful, False otherwise
        """
        ppcie_gpus, ppcie_switches = self.get_ppcie_capable_devices()
        all_devices = ppcie_gpus + ppcie_switches

        if not all_devices:
            logger.warning("No PPCIe-capable devices found")
            set_cc_mode_state_label(self.v1, self.node_name, mode)
            return True

        target_mode = "on" if mode == "ppcie" else "off"

        if self.ppcie_mode_is_set(all_devices, target_mode):
            logger.info(f"All devices already in PPCIe '{target_mode}'")
            set_cc_mode_state_label(self.v1, self.node_name, mode)
            return True

        if self.evict_operator_components:
            return self._set_ppcie_mode_with_eviction(all_devices, target_mode, mode)
        else:
            return self._set_ppcie_mode_direct(all_devices, target_mode, mode)

    def set_mode(self, mode: str) -> bool:
        """
        Set security mode based on label value.
        Routes to CC or PPCIe handlers based on mode.

        Args:
            mode: Desired mode ('on', 'off', 'devtools', 'ppcie')

        Returns:
            True if successful, False otherwise
        """
        if not mode:
            logger.info("No mode specified")
            return True

        if not self.validate_mode(mode):
            return False

        # Route based on mode
        if mode == 'ppcie':
            return self._handle_ppcie_mode()
        elif mode in ['on', 'devtools']:
            return self._handle_cc_mode(mode)
        else:  # mode == 'off'
            return self._handle_off_mode()

    def _handle_ppcie_mode(self) -> bool:
        """
        Enable PPCIe mode (disables CC first if needed).

        Returns:
            True if successful, False otherwise
        """
        # First disable CC on GPUs if enabled
        cc_gpus = self.get_cc_capable_gpus()
        for gpu in cc_gpus:
            try:
                if gpu.query_cc_mode() != "off":
                    logger.info("Disabling CC mode before enabling PPCIe")
                    if not self._set_cc_mode_internal("off"):
                        return False
                    break
            except Exception as e:
                logger.warning(f"Could not query CC mode on {gpu.bdf}: {e}")

        # Enable PPCIe
        return self.set_ppcie_mode("ppcie")

    def _handle_cc_mode(self, mode: str) -> bool:
        """
        Enable CC mode (disables PPCIe first if needed).

        Args:
            mode: CC mode to set ('on' or 'devtools')

        Returns:
            True if successful, False otherwise
        """
        # First disable PPCIe on all devices if enabled
        ppcie_gpus, ppcie_switches = self.get_ppcie_capable_devices()
        all_ppcie = ppcie_gpus + ppcie_switches

        for device in all_ppcie:
            try:
                if device.query_ppcie_mode() == "on":
                    logger.info("Disabling PPCIe mode before enabling CC")
                    if not self.set_ppcie_mode("off"):
                        return False
                    break
            except Exception as e:
                logger.warning(f"Could not query PPCIe mode on {device.bdf}: {e}")

        # Enable CC
        return self._set_cc_mode_internal(mode)

    def _handle_off_mode(self) -> bool:
        """
        Disable both CC and PPCIe modes.

        Returns:
            True if successful, False otherwise
        """
        success = True

        # Disable PPCIe first
        if not self.set_ppcie_mode("off"):
            logger.error("Failed to disable PPCIe mode")
            success = False

        # Then disable CC
        if not self._set_cc_mode_internal("off"):
            logger.error("Failed to disable CC mode")
            success = False

        if success:
            set_cc_mode_state_label(self.v1, self.node_name, "off")

        return success

    def get_node_cc_mode_label(self) -> None:
        """
        Get the current CC mode label from the node, updates local data.
        Quits if the get fails

        Returns:
            Nothing
        """
        try:
            node = self.v1.read_node(self.node_name)
            labels = node.metadata.labels or {}
            label_value = labels.get(CC_MODE_CONFIG_LABEL, '')
            resource_version = node.metadata.resource_version
            self.last_label = self.current_label
            self.current_label = label_value
            self.current_rv = resource_version
        except ApiException as e:
            logger.error(f"Failed to read node labels: {e}")
            sys.exit(1)
    
    def watch_and_apply(self) -> None:
        """
        Watch for changes to the node's CC mode label.
        
        This runs indefinitely and triggers CC mode changes when the label changes.
        
        """
        
        # Start with the initial value and version
        self.get_node_cc_mode_label()
        self.set_mode(self.with_default(self.current_label))
        # Create readiness file to indicate container is ready
        create_readiness_file()

        logger.info(f"Starting watch on node '{self.node_name}' for label '{CC_MODE_CONFIG_LABEL}' current_label: {self.current_label}")

        field_selector = f'metadata.name={self.node_name}'
        last_label_value = self.current_label
        consecutive_errors = 0
        
        while True:
            try:
                w = watch.Watch()

                # Build watch parameters
                watch_kwargs = {
                    'field_selector': field_selector,
                    'resource_version': self.current_rv,
                    'timeout_seconds': 300,  # 5 minute timeout
                }

                logger.info(f"Starting watch from ResourceVersion: {self.current_rv}")
                for event in w.stream(self.v1.list_node, **watch_kwargs):
                    event_type = event['type']
                    if event_type == 'ERROR':
                        # Error event from watch
                        logger.error(f"Watch error event: {event}")
                        consecutive_errors += 1
                        break  # Break inner loop to reconnect

                    # reset error count
                    consecutive_errors = 0
                    node = event['object']
                    if hasattr(node.metadata, 'resource_version') and node.metadata.resource_version:
                        self.current_rv = node.metadata.resource_version

                    if event['type'] in ('ADDED', 'MODIFIED'):
                        labels = node.metadata.labels or {}
                        self.current_label = labels.get(CC_MODE_CONFIG_LABEL, '')
                        # Only act if label actually changed
                        if self.current_label != last_label_value:
                            logger.info(
                                f"Label changed: '{last_label_value}' -> '{self.current_label}' "
                                f"(event: {event_type})"
                            )
                            last_label_value = self.current_label
                            self.set_mode(self.with_default(self.current_label))
                            continue
                        
            except ApiException as e:
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.error(
                        f"Watch failed {consecutive_errors} times consecutively, "
                        f"treating as fatal error"
                    )
                    raise RuntimeError(
                        f"Watch failed after {consecutive_errors} consecutive errors: {e}"
                    )

                if e.status == 410:
                    # ResourceVersion too old (etcd compacted)
                    logger.warning(
                        f"ResourceVersion {self.current_rv} is too old (410 Gone). "
                        f"Performing re-sync and starting fresh watch."
                    )
                    self.get_node_cc_mode_label()
                    if self.current_label != last_label_value:
                        logger.info(
                            f"Label changed: '{last_label_value}' -> '{self.current_label}' "
                        )
                        last_label_value = self.current_label
                        self.set_mode(self.with_default(self.current_label))
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)
    
    def with_default(self, label) -> str:
        """Apply default if label is empty"""
        if not label:
            logger.info(f"Applying default CC mode: {self.default_mode}")
            return self.default_mode
        return label

    def run(self) -> None:
        """Main entry point - start the CC manager."""
        self.watch_and_apply()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NVIDIA CC Manager For Kubernetes'
    )
    parser.add_argument(
        '--kubeconfig',
        default=os.environ.get('KUBECONFIG', ''),
        help='Absolute path to the kubeconfig file'
    )
    parser.add_argument(
        '--default-cc-mode', '-m',
        default=os.environ.get('DEFAULT_CC_MODE', 'on'),
        help='Security mode to be set by default when node label nvidia.com/cc.mode is not applied. '
             'Valid values: on, off, devtools, ppcie'
    )
    parser.add_argument(
        '--node-name',
        default=os.environ.get('NODE_NAME', ''),
        help='Kubernetes node name (default: $NODE_NAME)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('nvidia_gpu_tools').setLevel(logging.DEBUG)
    
    # Validate required environment variables
    if not args.node_name:
        logger.error("NODE_NAME environment variable must be set for k8s-cc-manager")
        sys.exit(1)
    
    # Create and run the manager
    try:
        manager = CCManager(
            node_name=args.node_name,
            default_mode=args.default_cc_mode
        )
        
        manager.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
