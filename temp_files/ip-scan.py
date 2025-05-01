from mac_vendor_lookup import MacLookup
import cv2
import socket
import struct
import requests
import nmap
import re
from getmac import get_mac_address

# ONVIF WS-Discovery Multicast Address
MULTICAST_GROUP = "239.255.255.250"
MULTICAST_PORT = 3702

def send_ws_discovery():
    """Sends an ONVIF WS-Discovery multicast request and listens for responses."""
    message = """<?xml version="1.0" encoding="UTF-8"?>
    <Envelope xmlns="http://www.w3.org/2003/05/soap-envelope">
        <Header>
            <wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
            uuid:e3a2f7a2-6c1c-11eb-8e9e-1f06da4401a1
            </wsa:MessageID>
            <wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">urn:schemas-xmlsoap-org:ws:2004:08:discovery</wsa:To>
            <wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
            http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe
            </wsa:Action>
        </Header>
        <Body>
            <Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
                <Types>dn:NetworkVideoTransmitter</Types>
            </Probe>
        </Body>
    </Envelope>"""

    # Send multicast UDP packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(3)
    
    try:
        sock.sendto(message.encode(), (MULTICAST_GROUP, MULTICAST_PORT))
        print("ONVIF Discovery request sent. Listening for responses...")
        
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                ip = addr[0]
                if ip not in discovered_ips:
                    discovered_ips.add(ip)
                    print(f"ONVIF Camera Found: {ip}")
            except socket.timeout:
                break
    except Exception as e:
        print(f"Error in ONVIF discovery: {e}")
    finally:
        sock.close()

def scan_network(subnet="192.168.1.0/24"):
    """Scans the network for devices with open camera-related ports."""
    nm = nmap.PortScanner()
    nm.scan(hosts=subnet, arguments="-p 80,443,554,8000,8080,8554 --open")
    
    for host in nm.all_hosts():
        open_ports = [port for port in nm[host]['tcp'] if nm[host]['tcp'][port]['state'] == 'open']
        # if any(p in open_ports for p in [554, 80, 443, 8000, 8080, 8554]):
        if all(p in open_ports for p in [80, 443, 554, 8000]):
            discovered_ips.add(host)
            print(f"Potential IP Camera Found: {host} (Ports: {open_ports})")

def check_http_headers(ip):
    """Checks HTTP headers for camera-specific identifiers."""
    try:
        response = requests.get(f"http://{ip}", timeout=2)
        server_header = response.headers.get("Server", "").lower()
        if any(keyword in server_header for keyword in ["boa", "goahead", "axis", "hikvision", "dahua", "sony", "uniview"]):
            print(f"[{ip}] Identified as Camera - Server Header: {server_header}")
            return True
    except requests.RequestException:
        pass
    return False

# def check_rtsp(ip):
#     """Checks if the IP has an active RTSP stream."""
#     rtsp_urls = [
#         f"rtsp://{ip}:554/",
#         f"rtsp://{ip}:8554/",
#         f"rtsp://{ip}:554/stream1",
#         f"rtsp://{ip}:554/live"
#     ]
    
#     for url in rtsp_urls:
#         try:
#             response = requests.get(url, timeout=2)
#             if response.status_code == 200:
#                 print(f"[{ip}] Active RTSP stream detected at {url}")
#                 return True
#         except requests.RequestException:
#             pass
#     return False


def check_rtsp(ip):
    rtsp_url = f"rtsp://admin:Lucky786$1@{ip}:554/Streaming/Channels/101?transport=tcp"  # Modify as needed for your camera
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        print(f"[{ip}] Active RTSP stream detected via OpenCV.")
        cap.release()
        return True
    else:
        print(f"[{ip}] No active RTSP stream detected via OpenCV.")
        return False


# def lookup_mac_vendor(ip):
#     """Retrieves MAC address and checks vendor."""
#     mac = get_mac_address(ip=ip)
#     if mac:
#         vendor = mac[:8]  # First 8 characters of MAC indicate vendor
#         return mac, vendor
#     return None, None

def lookup_mac_vendor(ip):
    """Retrieves MAC address and vendor details using the OUI lookup."""
    mac = get_mac_address(ip=ip)
    if mac:
        try:
            vendor = MacLookup().lookup(mac)
        except Exception as e:
            vendor = "Unknown"
        return mac, vendor
    return None, None

def discover_ip_cameras(subnet="192.168.1.0/24"):
    global discovered_ips
    discovered_ips = set()

    print("\n=== Step 1: Sending ONVIF Discovery Request ===")
    send_ws_discovery()

    print("\n=== Step 2: Scanning Network for Open Camera Ports ===")
    scan_network(subnet)

    potential_cameras = discovered_ips.copy()

    print("\n=== Step 3: Checking HTTP Headers for Camera Signatures ===")
    for ip in potential_cameras:
        if check_http_headers(ip):
            print(f"[{ip}] Confirmed via HTTP headers.")
        else:
            print(f"[{ip}] HTTP header check inconclusive.")

    print("\n=== Step 4: Checking for Active RTSP Streams ===")
    for ip in potential_cameras.copy():
        if check_rtsp(ip):
            print(f"[{ip}] Confirmed via RTSP check.")
        else:
            potential_cameras.discard(ip)
            print(f"[{ip}] RTSP check inconclusive.")

    print("\n=== Step 5: MAC Address Lookup ===")
    for ip in potential_cameras:
        mac, vendor = lookup_mac_vendor(ip)
        print(f"[{ip}] MAC Address: {mac}, Vendor: {vendor}")

    print("\n=== Final List of Potential IP Cameras ===")
    for ip in potential_cameras:
        print(f"Potential IP Camera: {ip}")


# Run discovery
discover_ip_cameras()

