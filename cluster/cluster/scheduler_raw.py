import subprocess
from .scheduler import register_scheduler, Scheduler
from .build_docker import build_docker
import cluster.utils as utils
from jinja2 import Template

@register_scheduler
class RawScheduler(Scheduler):
    """
    sschedule task to torchx
    """
    register_name = "raw"

    def submit_impl(self):
        """create job from template"""
        args = self.args
        utils.check_comment(args.comment)

        job_id = utils.generate_job_id()
        job_init_name = "{}-{}-{}".format(args.ldap, args.jobname, job_id)
        job_full_name = "{}-{}-{}-{}".format(args.ldap, args.jobname, job_id, args.comment)
        self.job_full_name = job_full_name
        print("===> Creating job [{}] ".format(job_full_name))

        print("===> Packing code to docker image")
        image_name = self.generate_code_image(args.jobname)
        print("code docker image: ", image_name)

        print("===> Generating yaml ")
        args.training_script_args = ' '.join(args.training_script_args)
        command = "python " + args.training_script + " " + args.training_script_args
        yaml_file = self.generate_yaml(args.yaml_template, job_init_name, job_full_name, image_name, command)
        print("    to ", yaml_file)

        print("===> Deploying job [{}]".format(job_full_name))
        kube_cmd = self.get_kube_cmd()
        deploy_cmd = "{} apply -f {}".format(kube_cmd, yaml_file)
        ret = subprocess.run(deploy_cmd, shell=True)
        if ret.returncode == 0:
            print("create job {} successfully!  You can manage your job by User Cmd now.".format(job_full_name))
        else:
            print("create job failed!")

        return job_full_name

    def show_cmd(self):
        """show user cmd after deploying a job"""

        cmd_file = "/workspace/cluster/deploy/job_cmd.template"
        with open(cmd_file, 'r') as f:
            cmds = f.read()
        
        render_dict = {
            "job_name": self.job_full_name,
            "pod_name": "<podname>",
            "namespace": self.args.namespace
        }

        new_cmds = Template(cmds).render(render_dict)

        print("************* User Cmd ************")
        print(new_cmds)
        print("***********************************")

    def generate_code_image(self, job_name):
        """generate code image"""
        args = self.args
        base_image="registry.aibee.cn/mla/ubuntu:18.04"
        image = build_docker(args.jobname, args.docker_file, args.base_image, args.dir_list)
        return image

    def generate_yaml(self, yaml_file, init_name, job_name, image_name, command):
        """generate deploy yaml from template"""
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        render_dict = {
            "namespace": self.args.namespace,
            "base_image": self.args.base_image,
            "init_name": init_name,
            "job_name": job_name,
            "init_image": image_name,
            "command": command
        }

        new_content = Template(yaml_content).render(render_dict)
        
        deploy_file = "/workspace/cluster/deploy/{}.yml".format(job_name)
        with open(deploy_file, 'w') as f:
            f.write(new_content)

        return deploy_file

    def get_kube_cmd(self):
        kube_cmd = self.args.kube_cmd
        kube_conf = self.args.kube_conf

        cmd = "{} --kubeconfig={} ".format(kube_cmd, kube_conf)
        return cmd