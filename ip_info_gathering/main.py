import pandas as pd
import ipinfo
import json
from tqdm import tqdm
import numpy as np


def get_info_info(ip):
    access_token = '2153516fb5135c'
    ip_dict = {}
    handler_ip = ipinfo.getHandler(access_token)
    print("[*] - Must find", len(ip))
    for i in tqdm(range(len(ip))):
        details = handler_ip.getDetails(ip[i])
        ip_dict[ip[i]] = details.all
        if i % 100 == 0 and i > 0:
            print("[!] - Printing", i)
            dump_ips(ip_dict, 'dataset/json_files/ip_'+str(i)+'.json')
            ip_dict = {}

def dump_ips(ip_dict, folder):
    j = json.dumps(ip_dict)
    f = open(folder, "w")
    f.write(j)
    f.close()


def main():
    dataset = 'dataset/paulo_darknet.csv'
    data = pd.read_csv(dataset , low_memory=False)
    dst_ip = np.array(data["Dst IP"])
    src_ip = np.array(data["Src IP"])
    ip_to_find = np.unique(np.concatenate((src_ip, dst_ip)))


    print(len(ip_to_find), len(dst_ip), len(src_ip))


    get_info_info(ip_to_find)


if __name__ == "__main__":
    main()
