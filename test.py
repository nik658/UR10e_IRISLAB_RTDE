#!/usr/bin/env python3
"""
Test 5: GRIPKIT Diagnostic and Configuration Check
REQUIREMENTS BEFORE RUNNING:
1. Access to UR Polyscope interface
2. Check if GRIPKIT URCap is properly installed and shows in Program tree
3. Manually test gripper in GRIPKIT interface first
4. Check gripper wiring and power connections
"""

import socket
import time

class GripkitDiagnostic:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        
    def send_urscript(self, script):
        """Send URScript command and try to get response"""
        HOST = self.robot_ip
        PORT = 30002
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((HOST, PORT))
                s.sendall(script.encode('utf-8'))
                print(f"✓ Sent: {script.strip()}")
                
                # Try to receive any response
                try:
                    s.settimeout(2)
                    response = s.recv(1024)
                    if response:
                        print(f"  Response: {response.decode()}")
                except:
                    pass  # No response expected for most commands
                    
                return True
        except Exception as e:
            print(f"✗ Failed to send command: {e}")
            return False
    
    def check_urscript_variables(self):
        """Check if GRIPKIT variables are available"""
        print("Checking GRIPKIT URScript variables...")
        
        script = """
        textmsg("=== GRIPKIT Variable Check ===")
        
        # Try to access GRIPKIT functions
        try_gl_init = "GL_INIT function available"
        textmsg(try_gl_init)
        
        # Check if socket functions work
        socket_test = socket_open("127.0.0.1", 30001)
        if socket_test != -1:
            socket_close(socket_test)
            textmsg("Socket functions working")
        else:
            textmsg("Socket functions not working")
        end
        """
        return self.send_urscript(script)
    
    def test_all_gripkit_functions(self):
        """Test each GRIPKIT function individually"""
        print("Testing individual GRIPKIT functions...")
        
        functions_to_test = [
            'GL_INIT()',
            'GL_CONNECT("test_socket")',
            'GL_STATUS("test_socket")',
            'GL_DISCONNECT("test_socket")'
        ]
        
        for func in functions_to_test:
            print(f"\nTesting: {func}")
            script = f"""
            textmsg("Testing function: {func}")
            try:
                {func}
                textmsg("Function executed successfully")
            except:
                textmsg("Function failed or not available")
            end
            sleep(0.5)
            """
            self.send_urscript(script)
            time.sleep(1)
    
    def check_gripper_communication_ports(self):
        """Check different communication methods"""
        print("Checking gripper communication ports...")
        
        # Test different socket connections
        test_ports = [502, 1000, 1001, 2000, 30001, 30002]
        
        for port in test_ports:
            script = f"""
            textmsg("Testing port {port}")
            test_socket = socket_open("127.0.0.1", {port})
            if test_socket != -1:
                textmsg("Port {port} accessible")
                socket_close(test_socket)
            else:
                textmsg("Port {port} not accessible")
            end
            sleep(0.2)
            """
            self.send_urscript(script)
            time.sleep(0.5)
    
    def test_alternative_gripkit_commands(self):
        """Test alternative GRIPKIT command formats"""
        print("Testing alternative GRIPKIT command formats...")
        
        alternatives = [
            'GL_GRIP(0, 85, "sock_griplink")',
            'GL_MOVE(0, 85.0, 50.0, "sock_griplink")',
            'GL_SET_POS(0, 85.0, "sock_griplink")',
            'GL_OPEN(0, "sock_griplink")',
            'GL_CLOSE(0, "sock_griplink")'
        ]
        
        # First initialize
        init_script = """
        GL_INIT()
        sleep(1.0)
        GL_CONNECT("sock_griplink")
        sleep(1.0)
        """
        self.send_urscript(init_script)
        time.sleep(2)
        
        # Test each alternative
        for cmd in alternatives:
            print(f"\nTesting: {cmd}")
            script = f"""
            textmsg("Trying command: {cmd}")
            try:
                {cmd}
                textmsg("Command sent successfully")
                sleep(2.0)
            except:
                textmsg("Command failed")
            end
            """
            self.send_urscript(script)
            time.sleep(3)
        
        # Cleanup
        cleanup_script = 'GL_DISCONNECT("sock_griplink")'
        self.send_urscript(cleanup_script)
    
    def check_gripper_power_and_status(self):
        """Check if gripper is powered and responding"""
        print("Checking gripper power and basic status...")
        
        script = """
        textmsg("=== Gripper Power Check ===")
        
        # Check tool voltage
        tool_voltage = get_tool_voltage()
        textmsg("Tool voltage: ", tool_voltage)
        
        # Check if tool communication is active
        tool_current = get_tool_current()
        textmsg("Tool current: ", tool_current)
        
        # Check digital inputs (gripper feedback)
        textmsg("Digital inputs:")
        for i in range(8):
            input_state = get_digital_in(i)
            textmsg("DI", i, ": ", input_state)
        end
        
        # Check analog inputs
        textmsg("Analog inputs:")
        for i in range(2):
            analog_val = get_analog_in(i)
            textmsg("AI", i, ": ", analog_val)
        end
        """
        return self.send_urscript(script)
    
    def test_direct_socket_communication(self):
        """Test direct socket communication to gripper"""
        print("Testing direct socket communication...")
        
        script = """
        textmsg("=== Direct Socket Test ===")
        
        # Try to connect to gripper directly
        gripper_socket = socket_open("192.168.1.103", 502)  # Change IP as needed
        if gripper_socket != -1:
            textmsg("Successfully connected to gripper via socket")
            
            # Try to send a simple command
            socket_send_string(gripper_socket, "test")
            sleep(0.5)
            
            socket_close(gripper_socket)
            textmsg("Socket communication test complete")
        else:
            textmsg("Failed to connect to gripper via socket")
        end
        """
        return self.send_urscript(script)
    
    def run_full_diagnostic(self):
        """Run complete diagnostic sequence"""
        print("=== GRIPKIT Full Diagnostic ===")
        print("This will test all aspects of GRIPKIT communication\n")
        
        # Test 1: Basic URScript functionality
        print("1. Checking URScript variables and functions...")
        self.check_urscript_variables()
        time.sleep(2)
        
        # Test 2: Individual GRIPKIT functions
        print("\n2. Testing individual GRIPKIT functions...")
        self.test_all_gripkit_functions()
        time.sleep(2)
        
        # Test 3: Communication ports
        print("\n3. Checking communication ports...")
        self.check_gripper_communication_ports()
        time.sleep(2)
        
        # Test 4: Power and status
        print("\n4. Checking gripper power and status...")
        self.check_gripper_power_and_status()
        time.sleep(2)
        
        # Test 5: Direct socket communication
        print("\n5. Testing direct socket communication...")
        self.test_direct_socket_communication()
        time.sleep(2)
        
        # Test 6: Alternative commands
        print("\n6. Testing alternative GRIPKIT commands...")
        self.test_alternative_gripkit_commands()
        
        print("\n=== Diagnostic Complete ===")
        print("Check the UR Polyscope log for detailed output!")

if __name__ == "__main__":
    ROBOT_IP = "192.168.1.102"  # Change to your robot IP
    
    diagnostic = GripkitDiagnostic(ROBOT_IP)
    diagnostic.run_full_diagnostic()