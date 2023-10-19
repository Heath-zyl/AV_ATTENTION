import subprocess
from .scheduler import register_scheduler, Scheduler
from .build_docker import build_docker
import cluster.utils as utils

@register_scheduler
class RayScheduler(Scheduler):
    """
    sschedule task to ray
    """
    register_name = "ray"

    def submit_impl(self):
        import asyncio
        import ray
        from ray.job_submission import JobSubmissionClient
        from ray.job_submission import JobStatus
        
        args = self.args
        utils.check_comment(args.comment)

        print(args.training_script)
        print(args.training_script_args)
        args.training_script_args = ' '.join(args.training_script_args)
        command = "python " + args.training_script + " " + args.training_script_args
        client = JobSubmissionClient("http://172.16.10.29:32002")        
        job_id = client.submit_job(
            job_id="crystal-tune-{}-{}-{}".format(args.jobname, utils.generate_job_id(), args.comment),
            entrypoint=command,
            runtime_env={
                "working_dir": "/workspace",
                "excludes": [".git", "./log_loop/", "./docs", "./cluster"]
            }
        )

        async def print_logs(client, job_name):
            async for lines in client.tail_job_logs(job_name):
                print(lines, end="")
        asyncio.get_event_loop().run_until_complete(print_logs(client, job_id))

    def show_cmd(self):
        return
