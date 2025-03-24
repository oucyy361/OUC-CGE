import pandas as pd
import numpy as np
import torch


class OutResults:

    def __init__(self, output_path):

        self.output_path = output_path
        self.datas = {}

    def output_(self):


        data = np.vstack(list(self.datas.values())).T
        col = self.datas.keys()

        df = pd.DataFrame(data, columns=col)
        if "csv" in self.output_path:
            df.to_csv(self.output_path, index=False)
        elif "xls" in self.output_path or "xlsx" in self.output_path:
            df.to_excel(self.output_path, index=False)

        print(f"Saved the results in {self.output_path}")



    def add_(self, kwargs):

        for k, v in kwargs.items():

            if isinstance(v, list):
                self._add_list(k, v)
            elif isinstance(v, np.ndarray):
                self._add_numpy(k, v)
            elif isinstance(v, torch.Tensor):
                self._add_tensor(k, v)

        # print(self.datas)

    def _add_list(self, k, v):

        if k not in self.datas.keys():
            self.datas[k] = np.array(v)

        else:

            last_list = self.datas[k].tolist()
            last_list.extend(v)
            self.datas[k] = np.array(last_list)

    def _add_tensor(self, k, v):
        v = v.numpy()

        self._add_numpy(k, v)

    def _add_numpy(self, k, v):

        if v.ndim == 2:
            v= np.squeeze(v)
        if v.ndim > 2:
            print("Something wrong!")

        if k not in self.datas.keys():
            self.datas[k] = v

        else:
            self.datas[k] = np.hstack([self.datas[k], v])


