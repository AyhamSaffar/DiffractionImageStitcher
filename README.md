# Diffraction Image Stitcher

A graphical app that stitches together microscope images and dynamically loads in each pixel's diffraction pattern when needed.

![image](https://user-images.githubusercontent.com/76114207/188330103-72580335-6ab2-4121-927c-e71a341bd239.png)

## Method

Data is read folloiwng this expected format:

Each micrograph is expected to have a uniqe data time id in the YYYYMMDD_hhmmss format (IE 20211117_154630)

For each micrograph the following files are exptected with the micrograph id in their file name

    .jpeg image           
    meta data .hdf file       must contain entries at 'metadata/magnification' and 'metadata/step_size(m)'
    image data .hdf file      must contain an array at 'Experiments/__unnamed__/data'

Computer vision feature detection is then used to stitch these images together.

This is done by shifting each image in 2D so that similar features overlap.

Finally diffraction images are only loaded into memory when displayed.

## Installation

Donwload the *Image_Stitcher_App.py* file and run.

Several libraries must be installed for the app to work. This can be done by running the following in your command prompt terminal:

    pip install h5py hyperspy dask numpy opencv-python PySimpleGUI matplotlib Pillow

In order to access all the functionality seperately from the included interface, copy all the code up to the '''Front End''' comment (~line 300) into your python file or jupyter notebook.

Then create an instance of the 'App' class and and call its methods in order to access all the created data.

This would look something like:

    data = App('C:\Users\My_Name\Desktop\data_folder')
    stitch_image = data.find_stitch_image(stitch_index=1)
    diffraction_patterns = data.find_diff_images(stitch_index=1, x=5, y=3)
 
All methods are clearly documented. 

## Support

If you encounter any bugs or are finding any part of the process difficult please do let me know by raising an issue on GitHub. I'll do my best to reply within 24 hours.

## Roadmap

This app now has all the functionality i wanted however i am open to adding new functionality on request.

If you have any ideas for new features or changes please raise an issue on GitHub.

Possible future features could include:

- Stitch image upsampling to achieve better than 1 pixel allignement accuracy
- Diffraction image colourbar with selectable colour map
- Automatic jpeg generation from diffraction data if no jpeg data is found

## Authors and Acknowledgment

This program was written by me Ayham Saffar at the request of the Imperial - Cambridge Perovskite research group.

Thank you so much to everyone in the group for all your encouragement and support.
