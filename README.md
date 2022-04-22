# Diffraction Image Stitcher

A graphical app that stitches together microscope images and dynamically loads in each pixel's diffraction pattern when needed.

![Diffraction Image Stitcher Demo Gif](https://media.giphy.com/media/UDU0L4m2JzESLGVdPf/giphy.gif)

## Method

Micrograph image data in the following format is read in.

    jpeg path            {file_path}\{id}\{id}_ibf.jpg
    meta data path       {file_path}\{id}.hdf
    image data path      {file_path}\{id}\{id}_data.hdf5

Computer vision feature detection is then used to stitch these images together.

Finally diffraction images are loaded into memory and displayed only when selected.

## Installation

Donwload the *Image_Stitcher_App.py* file and run.

Several libraries must be installed for the app to work. This can be done by running the following command in terminal:

    pip install h5py hyperspy dask numpy opencv-python PySimpleGUI matplotlib Pillow

## Support

If you encounter any bugs or are finding any part of the process difficult please do let me know by raising an issue on GitHub. I'll do my best to reply within 24 hours.

## Roadmap

This app is in an early stage of developement and i am very keen to add the following features:

- File browser than can load in all data no matter how its structured

- Diffraction image threshold slider

- Diffraction image colorbar selector

- Micrograph pixel position indicator

- Tabs to show multiple diffraction images for overlapping micrographs

- Tabs to show multiple seperate micrograph stitches when all micrographs cannot be combined into one stich

- An export button that saves the micrograph stitch as a png with a scalebar

If you have any ideas for new features or changes please also raise an issue on GitHub.

## Authors and Acknowledgment

This program was written by me Ayham Saffar at the request of the Imperial - Cambridge Perovskite research group.

Thank you so much to everyone in the group for all your encouragement and support.
