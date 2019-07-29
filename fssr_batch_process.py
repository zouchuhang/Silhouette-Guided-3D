import os

source = "./data/result_ply/"
target = "./data/result_ply_out/"

files = next(os.walk(source))[2]

#print(len(files))
#time.sleep(500)  
cnt = 1
for file_name in files:
    print(cnt)
    if os.path.isfile(target+file_name[:-4]+"-clean.ply"):
        cnt+=1
        continue
    fssr_cmd = "./mve/apps/fssrecon/fssrecon "+source+file_name+" "+target+file_name
    fssr_cmd2 = "./mve/apps/meshclean/meshclean "+target+file_name + " "+target+file_name[:-4]+"-clean.ply"
    os.system(fssr_cmd)
    os.system(fssr_cmd2)
    cnt += 1
    #break
