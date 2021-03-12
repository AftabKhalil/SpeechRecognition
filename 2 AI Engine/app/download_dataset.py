import os
import threading
import pip
import json

from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account	
	
bucket_name = 'dspd_aftabkhalil_bucket'
dataset_id = 'dspd_aftabkhalil_dataset'
project_id = 'myfirstproject-305412'

def run_query(query):
	query_job = bigquery_client.query(query)
	data = query_job.result()
	return data

def get_resource_types():
	query = f'SELECT type FROM {dataset_id}.{table_id} group by type;'
	result = run_query(query)
	resource_types = []
	for r in list(result):
		resource_types.append(r.get('type'))
	return resource_types
	
def get_resources(root, resource_type):
	query = (f'SELECT name FROM {dataset_id}.{table_id} '
             f'WHERE type = "{resource_type}" AND location LIKE "{root}/{resource_type}/%" '
             f'GROUP BY name')
	result = run_query(query)
	resource = []
	for r in list(result):
		resource.append(r.get('name'))
	return resource

def create_or_get_bucket(bucket_name):

	#Get already existsing buckets
	buckets = list(storage_client.list_buckets())
	
	#Check if required bucket already exists
	bucket = next((b for b in buckets if b.name == bucket_name), None)
	
	#If bucket already exists retuen it
	if(bucket != None):
		return bucket
	#Else create and return bucket
	else:
		bucket = storage_client.bucket(bucket_name)
		new_bucket = storage_client.create_bucket(bucket, location="us")
		return new_bucket

def download_blob(resource_full_path):
	blob = bucket.blob(resource_full_path)
	blob.download_to_filename(f'app/{resource_full_path}')
	
def download_dataset(force_download = False):	
	resource_types = get_resource_types()
	resource_types.sort()
			
	for resource_type in resource_types:
		folder = f'app/{root}/{resource_type}'
		if not os.path.exists(folder):
			os.makedirs(folder)
			
		remote_resources = get_resources(root, resource_type)
		for remote_resource in remote_resources:
			resource_full_path = f'{root}/{resource_type}/{remote_resource}'
			if(force_download or not os.path.exists(f'app/{resource_full_path}')):
				download_blob(resource_full_path)

def download_data(parm_root, parm_table_id):
	global root
	global table_id
	global storage_client
	global bigquery_client
	global bucket
	
	root = parm_root 
	table_id = parm_table_id
	
	with open('app/myfirstproject-305412-bd26f6fbb24b.json') as source:
		info = json.load(source)
	
	storage_credentials = service_account.Credentials.from_service_account_info(info)
	
	storage_client = storage.Client(project = project_id, credentials = storage_credentials)
	bigquery_client = bigquery.Client(project = project_id, credentials = storage_credentials)

	create_or_get_bucket(bucket_name)	
	bucket = storage_client.bucket(bucket_name)

	download_dataset()
	
	return "downloaded"