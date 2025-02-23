import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap, AsyncSniffer, wrpcap
from datetime import datetime


def extract_flow_features(pcap_file, output_excel, isvpn, burst_threshold=1.0):
    if not os.path.exists(pcap_file):
        print(f"[ERROR] PCAP file not found: {pcap_file}")
        return

    print(f"[DEBUG] Reading PCAP file: {pcap_file}")
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"[ERROR] Failed to read PCAP file: {e}")
        return

    print(f"[DEBUG] Total packets read: {len(packets)}")

    if isvpn:
        output_excel = output_excel.replace('.xlsx', '_vpn.xlsx')
        print(f"[INFO] VPN scenario detected; output file: {output_excel}")

    flows = {}
    for pkt in packets:
        if pkt.haslayer('IP'):
            ip_layer = pkt['IP']
            if pkt.haslayer('TCP'):
                protocol = 'TCP'
                sport = pkt['TCP'].sport
                dport = pkt['TCP'].dport
            elif pkt.haslayer('UDP'):
                protocol = 'UDP'
                sport = pkt['UDP'].sport
                dport = pkt['UDP'].dport
            else:
                continue
            key = (ip_layer.src, ip_layer.dst, sport, dport, protocol)
            flows.setdefault(key, []).append((pkt.time, len(pkt)))

    print(f"[DEBUG] Total flows identified: {len(flows)}")

    rows = []
    for key, pkt_info in flows.items():
        pkt_info.sort(key=lambda x: float(x[0]))
        times = [float(t) for t, l in pkt_info]
        sizes = [l for t, l in pkt_info]
        num_packets = len(times)
        duration = times[-1] - times[0] if num_packets > 1 else 0

        fiat = [float(times[i] - times[i - 1]) for i in range(1, num_packets)]
        min_fiat = min(fiat) if fiat else 0
        max_fiat = max(fiat) if fiat else 0
        mean_fiat = np.mean(fiat) if fiat else 0

        biat = []
        for i in range(1, num_packets):
            if sizes[i] > 0:
                biat.append(fiat[i - 1] / sizes[i])
        min_biat = min(biat) if biat else 0
        max_biat = max(biat) if biat else 0
        mean_biat = np.mean(biat) if biat else 0

        flowiat_list = []
        current_burst_end = times[0]
        for i in range(1, num_packets):
            gap = float(times[i] - times[i - 1])
            if gap < burst_threshold:
                current_burst_end = times[i]
            else:
                burst_gap = times[i] - current_burst_end
                flowiat_list.append(float(burst_gap))
                current_burst_end = times[i]
        if flowiat_list:
            min_flowiat = min(flowiat_list)
            max_flowiat = max(flowiat_list)
            mean_flowiat = np.mean(flowiat_list)
            std_flowiat = np.std(flowiat_list)
        else:
            min_flowiat = max_flowiat = mean_flowiat = std_flowiat = 0

        active_times = [gap for gap in fiat if gap < burst_threshold]
        idle_times = [gap for gap in fiat if gap >= burst_threshold]
        min_active = min(active_times) if active_times else 0
        mean_active = np.mean(active_times) if active_times else 0
        max_active = max(active_times) if active_times else 0
        std_active = np.std(active_times) if active_times else 0
        min_idle = min(idle_times) if idle_times else 0
        mean_idle = np.mean(idle_times) if idle_times else 0
        max_idle = max(idle_times) if idle_times else 0
        std_idle = np.std(idle_times) if idle_times else 0

        flowPktsPerSecond = num_packets / duration if duration > 0 else num_packets
        total_bytes = sum(sizes)
        flowBytesPerSecond = total_bytes / duration if duration > 0 else total_bytes

        row = {
            'src': key[0],
            'dst': key[1],
            'sport': key[2],
            'dport': key[3],
            'protocol': key[4],
            'min_fiat': min_fiat,
            'max_fiat': max_fiat,
            'mean_fiat': mean_fiat,
            'min_biat': min_biat,
            'max_biat': max_biat,
            'mean_biat': mean_biat,
            'min_flowiat': min_flowiat,
            'max_flowiat': max_flowiat,
            'mean_flowiat': mean_flowiat,
            'std_flowiat': std_flowiat,
            'min_active': min_active,
            'mean_active': mean_active,
            'max_active': max_active,
            'std_active': std_active,
            'min_idle': min_idle,
            'mean_idle': mean_idle,
            'max_idle': max_idle,
            'std_idle': std_idle,
            'flowPktsPerSecond': flowPktsPerSecond,
            'flowBytesPerSecond': flowBytesPerSecond,
            'duration': duration
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[DEBUG] Feature DataFrame shape: {df.shape}")

    try:
        df.to_excel(output_excel, index=False)
        print(f"[INFO] Excel file saved to {output_excel}")
    except Exception as e:
        print(f"[ERROR] Failed to write Excel file: {e}")


def main():
    proxy_ip = input("Enter the proxy IP address (or leave blank to auto-detect): ").strip()
    proxy_port = input("Enter the proxy port (or leave blank if not applicable): ").strip()

    if not proxy_ip:
        print("[INFO] Auto-detect not implemented, please provide a proxy IP.")
        return

    if proxy_port:
        bpf_filter = f"host {proxy_ip} and port {proxy_port}"
    else:
        bpf_filter = f"host {proxy_ip}"

    print(f"[INFO] Starting packet capture with filter: {bpf_filter}")
    print("[INFO] Type 'stop' to end capture.")

    sniffer = AsyncSniffer(filter=bpf_filter)
    sniffer.start()

    while True:
        command = input()
        if command.strip().lower() == "stop":
            break

    sniffer.stop()
    packets = sniffer.results

    if packets:
        print(f"[DEBUG] {len(packets)} packet(s) captured.")
        try:
            first_summary = packets[0].summary()
            print(f"[DEBUG] First packet summary: {first_summary}")
        except Exception as e:
            print(f"[ERROR] Failed to get packet summary: {e}")
    else:
        print("[DEBUG] No packets captured.")

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    pcap_file = f"vpndata/training-data-{now}.pcap"
    try:
        wrpcap(pcap_file, packets)
        print(f"[INFO] Saved captured packets to {pcap_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write pcap file: {e}")
        return

    output_excel = f"vpndata/training-data-{now}.xlsx"
    extract_flow_features(pcap_file, output_excel, False)
    print("[INFO] Flow extraction complete.")


if __name__ == '__main__':
    main()