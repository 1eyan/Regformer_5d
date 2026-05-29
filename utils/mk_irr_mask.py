import os
import numpy as np
import seisio as sio
import pandas as pd
import logging
import pdb
import h5py as h5
#from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
log = logging.getLogger("main")

#complete write by xd, 20260508

output_type = 'segy'   # default output h5 data, or 'sgy' for segy output.

def segy2h5(h5_file, data, group_name='1551', headers_df=None):
    """
    单个 SEG-Y 落盘到 H5，按 sort_keys 组织地震道。
    """
    with h5.File(h5_file, 'w', locking=False) as h5f:
        g = h5f.create_group(group_name)
        g.create_dataset('data', data=data, dtype='f',compression='gzip')
        g.create_dataset('sx', data=headers_df['shot_x'], dtype='f',compression='gzip')
        g.create_dataset('sy', data=headers_df['shot_y'], dtype='f',compression='gzip')
        g.create_dataset('rx', data=headers_df['recv_x'], dtype='f',compression='gzip')
        g.create_dataset('ry', data=headers_df['recv_y'], dtype='f',compression='gzip')
        g.create_dataset('offset', data=headers_df['offset'],dtype='f', compression='gzip')
        g.create_dataset('dt', data=headers_df['dt'], dtype='f',compression='gzip')
        g.create_dataset('t0', data=headers_df['t0'],dtype='f', compression='gzip')
        g.create_dataset('shot_line', data=headers_df['shot_line'], dtype='i',compression='gzip')
        g.create_dataset('shot_no', data=headers_df['shot_no'], dtype='i',compression='gzip')
        g.create_dataset('recv_line', data=headers_df['recv_line'], dtype='i',compression='gzip')
        g.create_dataset('recv_no', data=headers_df['recv_no'], dtype='i',compression='gzip')
        g.create_dataset('shot_stake', data=headers_df['shot_stake'], dtype='i',compression='gzip')
        g.create_dataset('recv_stake', data=headers_df['recv_stake'], dtype='i',compression='gzip')
        g.create_dataset('cmp', data=headers_df['cmp'], dtype='i',compression='gzip')
        g.create_dataset('cmp_line', data=headers_df['cmp_line'],dtype='i', compression='gzip')
        g.create_dataset('trace_idx', data=headers_df.index, dtype='q',compression='gzip')


def to_native_endian(df):
    # Convert all numeric columns

    # Convert ALL big-endian numeric columns (including unsigned ints)
    for col in pdinfo.select_dtypes(include='number').columns:
        if pdinfo[col].dtype.byteorder == '>':
            pdinfo[col] = pdinfo[col].astype(pdinfo[col].dtype.newbyteorder('='))
    #for col in df.select_dtypes(include=['int', 'float']).columns:
    #    if df[col].dtype.byteorder == '>':
    #        df[col] = df[col].astype(df[col].dtype.newbyteorder('='))

    # Convert index (single or MultiIndex)
    idx = df.index
    if hasattr(idx, 'dtype'):  # single index (Int64Index, etc.)
        if idx.dtype.byteorder == '>':
            df.index = idx.astype(idx.dtype.newbyteorder('='))
    elif hasattr(idx, 'levels'):  # MultiIndex
        new_levels = []
        for level in idx.levels:
            if level.dtype.byteorder == '>':
                new_levels.append(level.astype(level.dtype.newbyteorder('=')))
            else:
                new_levels.append(level)
        df.index = pd.MultiIndex(levels=new_levels, codes=idx.codes)
    return df

#datapath4="../004-sw06-Sj5-irr.sgy"
#datapath5="../004-sw06-Sj5-mask.sgy"
#datapath3="/hw6p/groupdata.new/procai/xd/reg5d_pku/test2604/train/004_sw13-label.sgy"
#datapath3="/hw6p/groupdata.new/procai/xd/reg5d_pku/test2604/train/004_sw13-label"
#datapath3="/hw6p/groupdata.new/procai/xd/reg5d_pku/test2604/004_sw09-select-4ms-filter.sgy"

#by xd, 2026-05-11
#Here for input segy label, and all other data, including mask, irr (h5 & segy) are all from
#this segy data 
#datapath3="/hw6p/groupdata.new/procai/xd/reg5d_pku/test2604/train/004_sw11-label.sgy"
datapath3="/data/liuqi/code/MAE/5d-transformer/gated_v35/data/new/A002-train-part1.sgy"
#output location, modified as your will
out_path = "/data/liuqi/code/MAE/5d-transformer/gated_v35/data/new/train" #modified bg czt0511  #location for output data 
os.makedirs(out_path,exist_ok=True)
outmp = datapath3.split('/')[-1]
outname=outmp.split('.')[0]
print(f'The output file name is {outname}')
segy_path=f'{out_path}{outname}-'
outmp = os.path.join(out_path,'h5/')
#outmp=os.makedirs(out_path+'h5/',exist_ok=True)
os.makedirs(outmp,exist_ok=True)
print(f'outmp = {outmp}')
h5_path=f"{outmp}{outname}-"  # h5 filename
print(f'segy_path = {segy_path},  h5_path={h5_path}')


#Attention, by  xd, 2026-05-11
#specifically for the 5D data,  2026-04-21, Tuesday.
#If test other data, please do not use this customized header
#jsons='./5ddata_header.json'   #for Zhaobiao data only
#jsons='./inmodel_header.json'  #same with in2model_header.json
jsons='/data/liuqi/code/MAE/5d-transformer/gated_v35/data/new/in2model_header.json'  #input model header, spedified for model data 
#model_json = 'model_header.json'
#outjson = 'model_header.json'
jsons2='/data/liuqi/code/MAE/5d-transformer/gated_v35/data/new/in2model_header.json'  #same with in2model_header.json

#sio_tmp = sio.input('testdata.su')
#sio_tmp = sio.input('testdata.pp',filetype='SGY')  #used for other common dataset
sio_tmp = sio.input(datapath3,filetype='SGY',thdef=jsons) # jsons customized for 5D header extraction

#numpy contains all segy data and headers
#good for small data, not for large data
#dataset = sio_tmp.read_dataset() 
headall = sio_tmp.read_all_headers()  #load all header for sorting 

#print(f'headall  = {headall}')

ntraces = sio_tmp.nt  # or sio.ntraces
nsamples = sio_tmp.ns  # or sio.nsamples
sampling_interval = sio_tmp.vsi   #dt

print(f'ntraces,  nsamples,  sampling_interval = {ntraces, nsamples, sampling_interval} .')
#
thstat = sio_tmp.log_thstat(traces=headall)
print(f'thstat  = {thstat}')


outhead = {}
#outhead['trace']=list(range(ntraces))
#outhead['trace_idx']=range(ntraces)
outhead['shot_line']=headall['sp_line']
outhead['shot_no']=headall['source_no']
outhead['recv_line']=headall['gp_line']
outhead['recv_no']=headall['trace_no']
# 读取坐标信息
#outhead['shot_stake']=headall['shot_stake']
#outhead['recv_stake']=headall['recv_stake']
outhead['shot_stake']=headall['sp_point']
outhead['recv_stake']=headall['gp_point']
outhead['cmp']=headall['cmp']
outhead['cmp_line']=headall['cmp_line']
outhead['offset']=headall['offset']
#outhead['sx'] = headall['sx']
#outhead['sy'] = headall['sy']
#outhead['rx'] = headall['gx']
#outhead['ry'] = headall['gy']
outhead['shot_x'] = headall['sp_x']
outhead['shot_y'] = headall['sp_y']
outhead['recv_x'] = headall['gp_x']
outhead['recv_y'] = headall['gp_y']
outhead['t0'] = headall['delrt']
outhead['dt'] = headall['dt']
outhead['scalar']=headall['scalel']
#outhead['delta']=headall['delta']

pdinfo = pd.DataFrame.from_dict(outhead)
pdinfo.index.name = 'trace_idx'

pdinfo = to_native_endian(pdinfo)

print("Index dtype:", pdinfo.index.dtype)
print("Any big-endian columns?", any(pdinfo[col].dtype.byteorder == '>' for col in pdinfo.columns))

big_cols = [col for col in pdinfo.columns if pdinfo[col].dtype.byteorder == '>']
print("Still big-endian:", big_cols)   # Should be empty []

print(pdinfo)


big_endian_cols = [col for col in pdinfo.columns
                   if pdinfo[col].dtype.byteorder == '>']
print("Big-endian columns:", big_endian_cols)
#for col in big_endian_cols:
    # This swaps bytes and sets the dtype to native byte order
#    pdinfo[col] = pdinfo[col].values.byteswap().newbyteorder('=')

# for col in big_endian_cols:
#     arr = pdinfo[col].values                    # get the underlying NumPy array
#     arr.byteswap(inplace=True)                  # swap the bytes in the original data
#     pdinfo[col] = arr.view(arr.dtype.newbyteorder('='))  # reinterpret as native byte order

for col in big_endian_cols:
    # Target dtype: same type but with native byte order ('=')
    native_dtype = pdinfo[col].dtype.newbyteorder('=')
    pdinfo[col] = pdinfo[col].astype(native_dtype)
#sort_keys = big_endian_cols  # your list of column names
#for key in sort_keys:
#    if pdinfo[key].dtype.byteorder == '>':
#        pdinfo[key] = pdinfo[key].values.byteswap().newbyteorder('=')

#scalar = pdinfo['scalar'][tmp_idx]
#scalar[scalar == 0] = 1

#pdinfo[['sx','sy','gx','gy']] = pdinfo[['sx','sy','gx','gy']].div(
#    pdinfo['scalar'].replace(0,1).abs(), axis=0
#)

cols = ['shot_x', 'shot_y', 'recv_x', 'recv_y']
scalar_safe = pdinfo['scalar'].replace(0, 1).abs()   # creates a writable Series
pdinfo[cols] = pdinfo[cols].div(scalar_safe, axis=0)

#sort_keys = ['recv_line', 'recv_stake', 'shot_line', 'shot_stake']
sort_keys = ['shot_line', 'shot_stake','recv_line', 'recv_stake' ]
pdinfo.sort_values(by=sort_keys,ascending=[True,True,True,True],na_position='last',inplace=True)
#tmp_idx = pdinfo['trace_idx'].to_numpy(dtype=np.intp)
tmp_idx = pdinfo.index.to_numpy(dtype=np.intp)

#is_sorted = True
#if tmp_idx != np.arange(ntraces):
#    print(f'data not sorted,  now sorting and kept index, new index = {tmp_idx}.')
#    is_sorted = False    #not sorted, now sorted again

#pdinfo[['shot_x','shot_y','recv_x','recv_y']] /= abs(scalar)

#by xd, 2026-05-05
#headall = headall[tmp_idx]  #resort head sequence for later segy dumping

group_fields = ['sp_line', 'sp_point']   # change as needed
sio_tmp.create_index(group_by=group_fields, sort_by=['gp_line','gp_point'])
#sio_tmp.create_index( group_by=['gp_line','gp_stake'], sort_by=['sp_line','sp_stake'])

# %%
nensembles = sio_tmp.nensembles        # number of ensembles, or sio.ne for short
ntraces_per_ensemble = sio_tmp.nte     # vector containing number of traces per ensemble
max_ntraces = sio_tmp.maxnte           # size (no. of traces) of largest ensemble
ensemble_keys = sio_tmp.ensemble_keys  # keys to identify the different ensembles

#log.info("Maximum number of traces within all the ensembles: %d", max_ntraces)
#print(f'\nIndexing ... , {nensembles} gathers,  maximum {max_ntraces} traces of largest ensembles, and the ensemble_keys are {ensemble_keys}')

#print(f'Finish creating  index now.')
uniq_lines = np.unique(headall['gp_line'])
nlins = len(uniq_lines)
uniq_stakes = np.unique(headall['gp_point'])
npnts = len(uniq_stakes)

#print("Unique gp_line values:", uniq_lines, " total ",len(uniq_lines), " points.")
#print("Unique gp_stake values:", uniq_stakes, " total ",len(uniq_stakes), "points.")

# 2. Generate your 2D boolean mask (shape = len(lines) x len(stakes))
#    For example, random mask:
#mask = np.random.random((len(uniq_lines), len(uniq_stakes))) < 0.3   # 30% True

#modify missing ratio here, by xd, 2026-05-11
missing_perc=0.3

#rand_array = np.random.rand(nlins, npnts)
#quanti = np.quantile(rand_array,missing_perc)

#del_stakes = np.random.permutation(uniq_stakes)[:int(npnts*0.5)]
#del_stakes = np.where(rand_array < quanti)
#mask = np.where(rand_array < quanti)
mask = np.random.random((len(uniq_lines), len(uniq_stakes))) < missing_perc   # 30% True

#load all data and sorted, by xd, 2026-05-11
all_data = sio_tmp.read_all_traces()[tmp_idx]   # structured array, shape (n_traces,)

# 3. Map each trace to indices (fast, O(n_traces * log(n_lines))
line_idx = np.searchsorted(uniq_lines, all_data['gp_line'])
stake_idx = np.searchsorted(uniq_stakes, all_data['gp_point'])

# Ensure all values are found (should be, if lines/stakes cover all data)
assert np.all(line_idx < len(uniq_lines)) and np.all(stake_idx < len(uniq_stakes))

print(f"Uniq  lines: {uniq_lines},  points:   {uniq_stakes}")

#final mask  is trace_mask
# 2. Create a 1D boolean mask for the traces using the 2D mask
trace_mask = mask[line_idx, stake_idx]   # shape = (n_traces,)


#Save label data here before generating missing data by xd 2026-05-11
#modified by xd, 2026-05-11
#if output_type == 'h5':
#print(f'Dumping {h5_path}label.h5 now ...')
print(f'Dumping {h5_path}label.h5 now ...')
segy2h5(f'{h5_path}new.h5', all_data['data'], group_name='1551', headers_df=pdinfo) #output h5 label data
#output segy lable with sorted sequence
#sout = sio.output(segy_path+'new.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", format=5, txtenc="ebcdic",thdef=jsons2)
sout = sio.output(segy_path+'new.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", txtenc="ebcdic",thdef=jsons2)
sout.init()
nwritten = sout.write_traces(traces=all_data) #use original headers with sorted order, by xd, 2026-05-11
print(f'Total write mask traces {nwritten}')
sout.finalize()

print(f'Missing traces ratio: {1.0 - np.sum(trace_mask)/ntraces}')
#save the mask array here, by xd, 2026-05-12
np.save(segy_path+'bool_mask_arr',trace_mask)

# 3. Apply the mask to the structured array
dataset_irr = all_data[trace_mask]  #irregular data sampling

#data interpolation, keep header, zeroed traces
all_data['data'][~trace_mask]=0.
dataset = all_data

# Convert to native and then use NumPy's unique on the structured array
sxy_arr = dataset_irr[['sp_east', 'sp_north']].copy()
sxy_arr = sxy_arr.astype(np.dtype([('sp_east', np.int32), ('sp_north', np.int32)]))
sxy = np.unique(sxy_arr)
sxy = sxy.view(np.int32).reshape(-1, 2)   # convert to 2D array of ints

gxy_arr = dataset_irr[['gp_east', 'gp_north']].copy()
gxy_arr = gxy_arr.astype(np.dtype([('gp_east', np.int32), ('gp_north', np.int32)]))
gxy = np.unique(gxy_arr)
gxy = gxy.view(np.int32).reshape(-1, 2)   # convert to 2D array of ints

#coordinates QC by xd, 2026-05-11
np.savetxt(segy_path+'shot_xy_cut.dat', sxy, delimiter='\t', fmt='%d')
np.savetxt(segy_path+'rcvs_xy_cut.dat', gxy, delimiter='\t', fmt='%d')

#data QC by xd, 2026-05-11
dataset_irr['data'][:10000].astype('float32').tofile(f'{segy_path}tmp_5dgather_irr_{nsamples}.bin')
dataset['data'][:10000].astype('float32').tofile(f'{segy_path}tmp_5dgather_mask_{nsamples}.bin')

#if output_type == 'segy':   #czt 0511 merge 'segy' mode and 'h5' mode
print(f'Dumping {h5_path}irr.h5 now ...')
mask_new = pdinfo[trace_mask]
segy2h5(h5_path+f'irr_{1-missing_perc}.h5', dataset_irr['data'], group_name='1551', headers_df=mask_new) #add missing prec in file_path  czt0511
    
print(f'Dumping {h5_path}mask.h5 now ...')
segy2h5(h5_path+f'mask_{1-missing_perc}.h5', dataset['data'], group_name='1551', headers_df=pdinfo)

print(f'Dumping {segy_path}+irr.sgy now ...')
#save segy file
#sout = sio.output(segy_path+'irr.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", format=5, txtenc="ebcdic",thdef=jsons2)
sout = sio.output(segy_path+f'irr_{1-missing_perc}.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", txtenc="ebcdic",thdef=jsons2)
sout.init()
nwritten = sout.write_traces(traces=dataset_irr)
print(f'Total write irr traces {nwritten}')
sout.finalize()

#sout = sio.output(segy_path+'mask.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", format=5, txtenc="ebcdic",thdef=jsons2)
sout = sio.output(segy_path+f'mask_{1-missing_perc}.sgy', ns=sio_tmp.ns, vsi=sio_tmp.vsi, endian=">", txtenc="ebcdic",thdef=jsons2)
sout.init()
nritten = sout.write_traces(traces=dataset)
print(f'Total write mask traces {nwritten}')
sout.finalize()
#else:
print("All data dumping out, program completed!")

quit()
os._exit(os.EX_OK)


