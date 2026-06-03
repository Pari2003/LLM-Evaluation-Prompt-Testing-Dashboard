import urllib.request, json
req = urllib.request.Request('https://api.github.com/repos/Pari2003/LLM-Evaluation-Prompt-Testing-Dashboard/actions/runs?per_page=1')
req.add_header('User-Agent', 'Mozilla/5.0')
with urllib.request.urlopen(req) as res:
    data = json.loads(res.read())
run_id = data['workflow_runs'][0]['id']
jobs_url = data['workflow_runs'][0]['jobs_url']

req2 = urllib.request.Request(jobs_url)
req2.add_header('User-Agent', 'Mozilla/5.0')
with urllib.request.urlopen(req2) as res2:
    jobs_data = json.loads(res2.read())

for job in jobs_data['jobs']:
    print(f"Job: {job['name']} - {job['conclusion']}")
    for step in job['steps']:
        if step['conclusion'] == 'failure':
            print(f"  Step failed: {step['name']}")
