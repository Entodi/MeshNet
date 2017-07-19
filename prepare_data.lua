slocal utils = require 'utils'

filename = 'brain_full_path.txt'
utils.prepare_data(filename, {'T1.npy', 'gm_wm.npy'}, 'brain_gm_wm.t7', 256, 256, 256)