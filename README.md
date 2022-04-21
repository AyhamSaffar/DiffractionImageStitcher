# Diffraction Image Stitcher

A graphical app that stitches together microscope images and dynamically loads in each pixel's diffraction pattern when needed

![Diffraction Image Stitcher Demo Gif](https://media.giphy.com/media/UDU0L4m2JzESLGVdPf/giphy.gif)

## Method

Image data in the following format is read in.

'code
     self.jpgpath = f'{file_path}\{id}\{id}_ibf.jpg'
     self.metadatapath = f'{file_path}\{id}.hdf'
     self.imdatapath = f'{file_path}\{id}\{id}_data.hdf5' 
'
