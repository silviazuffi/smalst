import os
import pickle as pkl
import numpy as np
from smpl_webuser.serialization import load_model
import pickle as pkl

model_dir = 'smalst/smpl_models/'

def align_smal_template_to_symmetry_axis(v):
    # These are the indexes of the points that are on the symmetry axis
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]

    v = v - np.mean(v)
    y = np.mean(v[I,1])
    v[:,1] = v[:,1] - y
    v[I,1] = 0
    sym_path = os.path.join(model_dir, 'symIdx.pkl')
    symIdx = pkl.load(open(sym_path))
    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[symIdx]] = np.array([1,-1,1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds

def load_smal_model(model_name='my_smpl_00781_4_all.pkl'):
    model_path = os.path.join(model_dir, model_name)

    model = load_model(model_path)
    v = align_smal_template_to_symmetry_axis(model.r.copy())

   
    return v, model.f

def get_horse_template(model_name='my_smpl_00781_4_all.pkl', data_name='my_smpl_data_00781_4_all.pkl'):

    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    nBetas = len(model.betas.r)
    data_path = os.path.join(model_dir, 'my_smpl_data_00781_4_all.pkl')
    data = pkl.load(open(data_path))
    # Select average zebra/horse
    betas = data['cluster_means'][2][:nBetas]
    model.betas[:] = betas
    v = model.r.copy()
    return v


