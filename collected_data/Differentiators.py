import numpy as np
import pandas as pd

#----- Differentiator class
class differentiators():
    def __init__(self,time_dec=1e9):
        self.time_dec=time_dec
        
        #------------- data is a numpy array with features as rows and time series as columns and timestep as the first column--------
        
    
    def process_timestamps(self):
        #---------- Timestamps processing------------
        self.num_timestamps=len(self.data[:,0])
        self.timestamps_processed=(self.data[:,0]-self.data[0,0])/self.time_dec
        self.dt_timestamps=self.timestamps_processed[1:self.num_timestamps]-self.timestamps_processed[0:self.num_timestamps-1]

        # #['FFD','BFD','CD','FIFD','robust_diff']
        # if method=='FFD':
        #     d=1
        # elif method=='BFD':
        #     d=1
        # elif method=='CD':
        #     d=1
        # elif method=='robust_diff': 
        #     self.d_timestamps,self.d_data=self.robust_differentiator()

    def robust_differentiator(self,data,order=5):                        # robust diff is 
        self.data=data
        self.process_timestamps()
        self.row_size,self.col_size=self.data.shape
        self.diff_data=np.zeros_like(self.data)
        print(f"Row size: {self.row_size}, Column size: {self.col_size}\n")
        for col in range(0,self.col_size):
            for row in range(0,self.row_size-1):
                # if row<2:
                #     self.diff_data[row,1:]=(self.data[row+1,1:]-self.data[row,1:])/self.dt_timestamps[row]
                # elif row<self.row_size-2:
                #     self.diff_data[row,1:]=((2*(self.data[row+1,1:]-self.data[row-1,1:]))+(self.data[row+2,1:]-self.data[row-2,1:]))/(8*self.dt_timestamps[row])
                # else:
                #     self.diff_data[row,1:]=(self.data[row,1:]-self.data[row-1,1:])/self.dt_timestamps[row]

                if order==5:
                    if row<2:
                        self.diff_data[row,1:]=(self.data[row+1,1:]-self.data[row,1:])/self.dt_timestamps[row]
                    elif row<self.row_size-2:
                        self.diff_data[row,1:]=((2*(self.data[row+1,1:]-self.data[row-1,1:]))+(self.data[row+2,1:]-self.data[row-2,1:]))/(8*self.dt_timestamps[row])
                    else:
                        self.diff_data[row,1:]=(self.data[row,1:]-self.data[row-1,1:])/self.dt_timestamps[row]
                elif order==7:
                    if row<3:
                        self.diff_data[row,1:]=(self.data[row+1,1:]-self.data[row,1:])/self.dt_timestamps[row]
                    elif row<self.row_size-3:
                        self.diff_data[row,1:]=((5*(self.data[row+1,1:]-self.data[row-1,1:]))+4*(self.data[row+2,1:]-self.data[row-2,1:])+(self.data[row+3,1:]-self.data[row-3,1:]))/(32*self.dt_timestamps[row])
                    else:
                        self.diff_data[row,1:]=(self.data[row,1:]-self.data[row-1,1:])/self.dt_timestamps[row]
                elif order==9:
                    if row<4:
                        self.diff_data[row,1:]=(self.data[row+1,1:]-self.data[row,1:])/self.dt_timestamps[row]
                    elif row<self.row_size-4:
                        self.diff_data[row,1:]=((14*(self.data[row+1,1:]-self.data[row-1,1:]))+14*(self.data[row+2,1:]-self.data[row-2,1:])+6*(self.data[row+3,1:]-self.data[row-3,1:])+(self.data[row+3,1:]-self.data[row-3,1:]))/(128*self.dt_timestamps[row])
                    else:
                        self.diff_data[row,1:]=(self.data[row,1:]-self.data[row-1,1:])/self.dt_timestamps[row]
        self.diff_data[:,0]=self.data[:,0]

        return self.timestamps_processed,self.diff_data
        #---------------------  The following content is for rough -----------------------
            # self.half_side_terms=N//2
            # self.indices=np.array([i for i in range(-self.half_side_terms,self.half_side_terms+1)])
            # self.indices=np.delete(self.indices,self.half_side_terms)

            # for row in range(0,self.row_size):
        #---------------------------------------------------------------------------------

        

        
