from morflowgenesis.utils.create_temporary_dask_cluster import encode_dict_to_json_base64
import os


args = {
    "cluster_class": 'distributed.LocalCluster',
    "cluster_kwargs":{
        "memory_limit": "5Gi",
        "resources":{
            'cpu': 1
        }
    },
    "adapt_kwargs":{
        'minimum': 10,
        'maximum': 64
    }
}


def main():
    base_64 = encode_dict_to_json_base64(args)
    print(f'export DASK_CLUSTER={base_64}')

if __name__ == '__main__':
    main()





