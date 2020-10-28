import json
import pandas as pd

json_str = '{"country":"Netherlands","dma_code":"0",\
            "timezone":"Europe\/Amsterdam","area_code":"0",\
            "ip":"46.19.37.108","asn":"AS196752","continent_code":"EU",\
            "isp":"Tilaa V.O.F.","longitude":5.75,"latitude":52.5,\
            "country_code":"NL","country_code3":"NLD"}'

data = json.loads(json_str)
print("Country: {}\n".format(data["country"]))
data["country"] = "Brazil"
print('json.dumps: {}\n'.format(json.dumps(data)))

data = pd.read_json(json_str, typ='series')
print("{} > Series\n".format(data))

data["country"] = "Brazil"
print("{} > New Series\n".format(data.to_json()))

# import numpy as np
# import pandas as pd

# np.random.seed(42)
# a = np.random.randn(365, 4)

# tmp = "pytable_df_demo2.h5"
# with pd.io.pytables.HDFStore(tmp) as store:
#     print(store)
#     df = pd.DataFrame(a)
#     store['df'] = df

#     print("Get", store.get('df').shape)
#     print("Lookup", store['df'].shape)
#     print("Dotted", store.df.shape)

#     del store['df']
#     print("After del\n", store)

#     print("Before close", store.is_open)
#     print("After close", store.is_open)

#     df.to_hdf('test.h5', 'data', format='table')
#     print(pd.read_hdf('test.h5', 'data', where=['index>363']))

# import numpy as np
# import tables
# from os.path import getsize

# np.random.seed(42)
# a = np.random.randn(365, 4)

# tmp = "pytable_demo.h5"
# with tables.open_file(tmp, mode='w') as h5:
#     root = h5.root
#     h5.create_array(root, "array", a)
#     h5.close()

#     h5 = tables.open_file(tmp, "r")
#     print('使用PyTables存储的文件大小: {}'.format(getsize(tmp)))

#     for node in h5.root:
#         b = node.read()
#         print(type(b), b.shape)


# import h5py
# import numpy as np


# def main():
#     # 创建一个HDF5文件
#     with h5py.File("h5py_example.hdf5", "w") as f:  # mode = {'w', 'r', 'a'}
#         # 在根目录'/'下创建两个组
#         g1 = f.create_group("bar1")
#         g2 = f.create_group("bar2")

#         # 在根目录下创建一个dataset
#         d = f.create_dataset("dset", data=np.arange(16).reshape([4, 4]))

#         # 对数据集dset添加两个Attributes
#         d.attrs["myAttr1"] = [100, 200]
#         d.attrs["myAttr2"] = "Hello, world!"

#         # 在"car1"分组下创建一个group和一个dataset
#         c1 = g1.create_group("car1")
#         print('group c1:\n', c1)
#         d1 = g1.create_dataset("dset1", data=np.arange(10))
#         print('dataset d1:\n', d1)

#         # 在"car2"分组下创建一个group和一个dataset
#         c2 = g2.create_group("car2")
#         print('group c2:\n', c2)
#         d2 = g2.create_dataset("dset2", data=np.arange(10))
#         print('dataset d2:\n', d2)

#     # 读取HDF5文件
#     with h5py.File("h5py_example.hdf5", "r") as f:  # mode = {'w', 'r', 'a'}
#         # 打印根分组'/'下的group和dataset名称
#         print(f.filename, ":")
#         print([key for key in f.keys()], "\n")

#         # 读取根group'/'下的数据集'dset'
#         d = f["dset"]

#         # 打印数据集'dset'的数据
#         print(d.name, ":")
#         print(d[:])

#         # 打印数据集'dset'的属性（元数据）
#         for key in d.attrs.keys():
#             print(key, ":", d.attrs[key])

#         # 读取'bar1'分组
#         g = f["bar1"]

#         # 打印分组'bar1'下的group和dataset名称
#         print([key for key in g.keys()])

#         # 三种访问数据集dset1的方法
#         print(f["/bar1/dset1"][:])  # 1. 绝对路径

#         print(f["bar1"]["dset1"][:])  # 2. 相对路径: file[][]

#         print(g['dset1'][:])  # 3. 相对路径: group[]

#         # 删除一个数据集（需在mode为可写状态下）
#         # del g["dset1"]


# import numpy as np
# import pandas as pd
# from os.path import getsize

# np.random.seed(42)
# a = np.random.randn(365, 4)

# tmp = 'code/savetext.csv'
# # tmp = NamedTemporaryFile()  # 可能会报PermissionError
# np.savetxt(tmp, a, delimiter=',')
# print("使用savetxt函数的CSV文件大小: ", getsize(tmp))

# tmp = 'code/save'
# np.save(tmp, a)
# tmp = '{}.npy'.format(tmp)
# loaded = np.load(tmp)
# print("Shape: {}".format(loaded.shape))
# print("使用save函数的npy文件大小: ", getsize(tmp))

# df = pd.DataFrame(a)
# tmp = 'code/pickle'
# df.to_pickle(tmp)
# print("使用to_pickle函数的npy文件大小: ", getsize(tmp))
# print("DF from pickle: \n", pd.read_pickle(tmp))

# import numpy as np
# import pandas as pd

# np.random.seed(42)

# a = np.random.randn(3, 4)
# a[2][2] = np.nan
# print(a)
# np.savetxt('code/np.csv', a, fmt='%.2f', delimiter=',', header=" #1, #2,  #3,  #4")
# df = pd.DataFrame(a)
# print(df)
# df.to_csv('code/pd.csv', float_format='%.2f', na_rep="NAN!")
