import time

def check_comment(comment) :
    assert comment is not None, "comment should not be None"
    assert len(comment) <=10 , "comment length should less equal 10"
    assert comment.islower() and comment.isalnum(), "comment should contains only numbers and lowercase letters"


def generate_job_id_simple():
    """generate unique job id by rule"""
    job_id = time.strftime("%m%d-%H%M", time.localtime(time.time()))
    return job_id

def generate_job_id():
    """generate unique job id by rule"""
    job_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    return job_id
