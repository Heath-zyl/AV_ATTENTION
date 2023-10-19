import os 
import time 

def gen_dockerfile(docker_file, base_image, dir_list):
    with open(docker_file, "w") as file_handle:
        # print(docker_file)
        file_handle.write("FROM {}\n".format(base_image))
        file_handle.write("RUN mkdir -p /workspace\n")
        file_handle.write("WORKDIR /workspace\n")
        for copy_dir in dir_list:
            if os.path.exists(copy_dir):
                dst_path = os.path.join("/workspace", copy_dir)
                copy_cmd = "COPY ./" + copy_dir + " " + dst_path + "\n"
                file_handle.write(copy_cmd)
        
        file_list = os.listdir(".")
        for file_name in file_list:
            if '.gitignore' in file_name:
                continue
            if os.path.isfile(file_name):
                file_handle.write("COPY ./%s /workspace\n"%file_name)

        # file_handle.write('RUN export PYTHONPATH=/workspace:$PYTHONPATH\n')


def build_docker(job_name, docker_file, base_image, dir_list):
    if isinstance(dir_list, str):
        dir_list = dir_list.split(",")
    gen_dockerfile(docker_file, base_image, dir_list)
    timeStruct = time.localtime()
    docker_tag = "registry.aibee.cn/aibee/aicluster/"+job_name+"-crystal-code:"+time.strftime('%Y-%m-%d_%H.%M.%S', timeStruct)

    command = 'docker build -t %s . -f %s'%(docker_tag, "/workspace/cluster/docker/Dockerfile")
    print(command)
    if os.system(command):
        raise "build docker error !"

    command = 'docker login registry.aibee.cn -u mla -p qwe123$%^'
    if os.system(command):
        raise "docker login error !"

    command = 'docker push %s' % docker_tag
    if os.system(command):
        raise "push docker error !"

    return docker_tag

if __name__ == '__main__':
    build_docker("test")