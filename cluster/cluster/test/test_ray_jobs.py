import requests
import json
import time
from ray.job_submission import JobStatus

resp = requests.post(
    "http://172.16.10.29:32002/api/jobs/",
    json={
        "entrypoint": "echo hello",
        "runtime_env": {},
        "job_id": None,
        "metadata": {"job_submission_id": "123"}
    }
)

rst = json.loads(resp.text)
job_id = rst["job_id"]

start = time.time()
while time.time() - start <= 10:
    resp = requests.get(
        "http://172.16.10.29:32002/api/jobs/{}".format(job_id)
    )
    rst = json.loads(resp.text)
    status = rst["status"]
    print(f"job {job_id} status: {status}")
    if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
        break
    time.sleep(1)


