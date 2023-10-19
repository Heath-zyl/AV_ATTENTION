import subprocess
from .scheduler import register_scheduler, Scheduler
from .build_docker import build_docker
import cluster.utils as utils
from jinja2 import Template
from torchx.components.dist import ddp as dist_ddp
from torchx import schedulers, specs
from torchx.schedulers.kubernetes_scheduler import (
    create_scheduler,
    KubernetesScheduler,
    KubernetesOpts
)

@register_scheduler
class TorchxScheduler(Scheduler):
    """
    sschedule task to torchx
    """
    register_name = "torchx"

    def submit_impl(self):
        args = self.args
        utils.check_comment(args.comment)

        job_id = utils.generate_job_id_simple()
        job_name = "{}-{}-{}-{}".format(args.ldap, args.jobname, job_id, args.comment)
        print("===> Creating dist [{}] ".format(job_name))

        print("===> Packing code to docker image")
        image = build_docker(job_name, args.docker_file, args.base_image, args.dir_list)
        print("code docker image: ", image)

        app_def = dist_ddp(
            *args.training_script_args,
            script = args.training_script,
            image = image,
            name = job_name,
            cpu = args.ncpu_per_node,
            gpu = args.ngpu_per_node,
            memMB = args.mbmem_per_node,
            selectors = args.node_selector,
            tolerations = args.tolerations,
            j = f'{args.nnodes}x{args.nproc_per_node}',
            env = args.env,
            max_retries = 0,
            mounts = args.mounts.split(",")
        )
        print("===> Torchx dist app def:")
        print(app_def)
        scheduler = create_scheduler("test")
        cfg = KubernetesOpts(
            {
                "queue": "default", 
                "namespace": args.namespace,
            }
        )

        info = scheduler._submit_dryrun(app_def, cfg)
        print(info)
        
        try:
            id = scheduler.schedule(info)
        except Exception as e:
            print("===> Torchx failed info:")
            print(info)
            print(e)
            print("create distributed job {} failed !".format(job_name))
            exit(-1)
        
        print("torchx scheduled id : {}".format(id))
        print("create distributed job {} successfully!  You can manage your job by User Cmd now.".format(job_name))

    def show_cmd(self):
        """show user cmd after deploying a dist"""

        cmd_file = "/workspace/cluster/deploy/dist_cmd.template"
        with open(cmd_file, 'r') as f:
            cmds = f.read()
        
        render_dict = {
            "job_name": "<jobname>",
            "pod_name": "<podname>",
            "namespace": self.args.namespace
        }

        new_cmds = Template(cmds).render(render_dict)

        print("************* User Cmd ************")
        print(new_cmds)
        print("***********************************")
