import os

source = "/data/czou4/result_4096_ply_no2d_n6n3/"
target = "/data/czou4/result_4096_ply_out_no2d_n6n3/"

files = next(os.walk(source))[2]

#print(len(files))
#time.sleep(500)  
cnt = 1
for file_name in files:
    print(cnt)
    if cnt<200000: #or cnt > 150000:
        cnt+=1
        continue
    if os.path.isfile(target+file_name[:-4]+"-clean.ply"):
        cnt+=1
        continue
    fssr_cmd = "../mve/apps/fssrecon/fssrecon "+source+file_name+" "+target+file_name
    fssr_cmd2 = "../mve/apps/meshclean/meshclean "+target+file_name + " "+target+file_name[:-4]+"-clean.ply"
    os.system(fssr_cmd)
    os.system(fssr_cmd2)
    cnt += 1
    #break
