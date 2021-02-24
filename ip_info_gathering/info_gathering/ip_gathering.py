import ipinfo
import json
from tqdm import tqdm

class IPGathering:
    def __int__(self):
        self.access_token = '2153516fb5135c'
        super(IPGathering, self).__int__()


    def get_info_info(self, ip):
        self.ip_dict = {}
        handler_ip = ipinfo.getHandler(self.access_token)
        print("[*] - Must find", len(ip))
        for i in tqdm(range(len(ip))):
            details = handler_ip.getDetails(ip[i])
            self.ip_dict[ip[i]] = details.all
            if i % 100 == 0 and i > 0:
                print("[!] - Printing", i)
                self.dump_ips(self.ip_dict, 'dataset/json_files/ip_'+str(i)+'.json')
                self.ip_dict = {}

    def dump_ips(self, ip_dict, folder):
        j = json.dumps(ip_dict)
        f = open(folder, "w")
        f.write(j)
        f.close()

