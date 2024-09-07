import h5py

# Open the HDF5 file
with h5py.File('model_saved.h5', 'r') as f:
    # List all groups and datasets in the file
    print("Groups and Datasets in HDF5 file:")
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    {}: {}".format(key, val))

    f.visititems(print_attrs)