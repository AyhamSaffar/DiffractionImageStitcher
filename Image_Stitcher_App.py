'''Dependencies'''


import h5py as h5
import hyperspy.api as hs
import dask.array as da
import os
import numpy as np
import cv2
import PySimpleGUI as sg
import io
import time
from matplotlib import pyplot as plt
from PIL import Image

def fig2data(fig):
    '''coverts a numpy array into a PIL image so that it can be displayed in PySimpleGUI'''
    img = Image.fromarray(fig)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


'''Back End'''


class Micrograph():
    '''
    Stores key micrograph parameters as attributes
    '''
    def __init__(self, file_path, id):
        self.id = id
        self.tiffpath = f'{file_path}\{id}\{id}_ibf.tiff'
        self.jpgpath = f'{file_path}\{id}\{id}_ibf.jpg'
        self.metadatapath = f'{file_path}\{id}.hdf'
        self.imdatapath = f'{file_path}\{id}\{id}_data.hdf5' 
        
        self.imarray = cv2.imread(self.jpgpath, cv2.IMREAD_GRAYSCALE)
        self.find_features()
        self.pysize, self.pxsize = np.shape(self.imarray) #size of each axis in pixels
        self.scale_factor = 1 #origional size * scale_factor = current size
        # self.edgedata = cv2.Canny(self.imarray, 255, 200)
        
        hdf = h5.File(self.metadatapath, 'r')
        self.mag = int(np.array(hdf['metadata/magnification'])) #pixel len / real len
        hdf.close()

        hdf = h5.File(self.imdatapath, 'r')
        #loading diffraction data into storage
        hdf_data = da.from_array(hdf['Experiments/__unnamed__/data'], chunks=(1,1,515,515))
        #creating arrays from data that are only loaded into memory as needed
        self.difdata = hs.signals.Signal2D(hdf_data).as_lazy().data

    def find_features(self):
        '''finds the key points for the given image using SIFT features (implemented using opencv2)'''
        sift = cv2.SIFT_create()
        self.features, self.descriptors = sift.detectAndCompute(self.imarray, None)
        self.cxpos = self.cypos = None #computed x an y positions of each image


class Stitch():
    '''Object containing stitched together Image objects and stitching methods'''
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.ids = self.find_ims()
        self.ims = [Micrograph(file_path, id) for id in self.ids]
        self.global_mag = self.find_most_common_mag()
    
    def find_ims(self):
        '''returns all the ims present in the selected file path'''
        return sorted([f for f in os.listdir(self.file_path) if '.' not in f])

    def find_most_common_mag(self):
        '''returns the most common magnification in self.ims'''
        mags = list([im.mag for im in self.ims])
        mag_counts = {mags.count(unique_mag) : unique_mag for unique_mag in set(mags)}
        return mag_counts[max(mag_counts.keys())]

    def normalise_im_sizes(self):
        '''scales all image sizes to 'global_mag' size using cubic spline interpolation'''
        for im in self.ims:
            if im.mag != self.global_mag:
                im.scale_factor = self.global_mag/im.mag
                im.pysize, im.pxsize = (int(im.pysize*im.scale_factor), int(im.pxsize*im.scale_factor))
                im.imarray = cv2.resize(im.imarray, (im.pysize, im.pxsize), interpolation=cv2.INTER_CUBIC)
                im.find_features()
                im.mag = self.global_mag
    
    def best_matches(self, im1_index, im2_index):
        '''returns the 5 best matches for any two micrograph ims in descending order'''
        im1, im2 = self.ims[im1_index], self.ims[im2_index]
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(im1.descriptors, im2.descriptors)
        return sorted(matches, key=lambda x:x.distance)[:5]

    def show_matches(self, im1_index, im2_index):
        '''displays the 5 best matches for any two micrograph ims'''
        best_matches = self.best_matches(im1_index, im2_index)
        result = cv2.drawMatches(self.ims[im1_index].imarray, self.ims[im1_index].features,
                                 self.ims[im2_index].imarray, self.ims[im2_index].features,
                                 best_matches, None, flags=2)
        return result

    def best_offsets(self, im1_index, im2_index):
        '''returns the 5 best offsets [(xoff, yoff)] for the im2 relative to im 1'''
        best_matches = self.best_matches(im1_index, im2_index)
        im1, im2 = self.ims[im1_index], self.ims[im2_index]
        offsets = []
        for match in best_matches:
            feature1, feature2 = im1.features[match.queryIdx], im2.features[match.trainIdx]
            xoff, yoff = int(feature1.pt[0]-feature2.pt[0]), int(feature1.pt[1]-feature2.pt[1])
            offsets.append((xoff, yoff))
        return np.array(offsets)

    def check_matches(self, im1_index, im2_index):
        '''returns True if the top matches for any two micrograph ims consistently return the same offset. Otherwise returns False'''
        best_offsets = self.best_offsets(im1_index,im2_index)
        return np.std(best_offsets[:,0]) < 2 and np.std(best_offsets[:,1]) < 2

    def find_cpos(self, im1_index, im2_index):
        '''finds the calculated offset of the second image compared to the first and uses this to find the calculated position'''
        xoff, yoff = self.best_offsets(im1_index, im2_index)[0]
        self.ims[im2_index].cxpos, self.ims[im2_index].cypos = self.ims[im1_index].cxpos + xoff, self.ims[im1_index].cypos + yoff

    def normalize_cpositions(self):
        '''normalises the calculated positions of images so the bottom left is at 0,0'''
        miny, minx = min([im.cypos for im in self.ims if im.cypos!=None]), min([im.cxpos for im in self.ims if im.cxpos!=None])
        for im in self.ims:
            if im.cxpos == None:
                continue
            im.cypos -= miny
            im.cxpos -= minx

    def draw_stitch(self):
        '''returns an array of the stitched image'''
        size = [max([im.cypos for im in self.ims if im.cypos!=None]) + 255, max([im.cxpos for im in self.ims if im.cxpos!=None]) + 255]
        canvas = np.full(size, 255, dtype=np.uint8)
        for im in self.ims:
            if im.cxpos == None:
                continue
            canvas[im.cypos:im.cypos+im.pysize, im.cxpos:im.cxpos+im.pxsize] = im.imarray
        return canvas

    def display_stitch(self, fig):
        '''displays the stitched array in the stitch window'''
        fig_height, fig_width = np.shape(fig)
        window['-STITCH-'].set_size(size=(fig_width, fig_height))
        window['-STITCH-'].change_coordinates((0,fig_height), (fig_width, 0))
        window['-STITCH-'].draw_image(data=fig2data(fig), location=(0,0))

    def display_diff(self, x, y):
        '''
        displays the diffraction image of the pixel at (x,y) in the diffraction box

        for scaled images the diffraction pattern of the nearest pixel in the unscaled image is used
        '''
        for im in self.ims:
            if im.cxpos != None:
                if im.cypos <= y < im.cypos+im.pysize and im.cxpos <= x < im.cxpos+im.pxsize:
                    dify, difx = (y-im.cypos)//im.scale_factor, (x-im.cxpos)//im.scale_factor
                    fig = im.difdata[dify,difx].compute()
                    fig = fig*(255//np.amax(fig))
                    window['-DIFF-'].update(data=fig2data(fig))
                    break
    
    def backup_stitch(self, im1_index, im2_index):
        '''
        Creates a popup when stitching fails displaying why the stitch failed and backup options.
        
        From here you can either skip the image or try stitching it to all the other images.
        '''
        layout = [
            [sg.Graph((800,600), (0,0), (100,100), key='-ERROR-')],
            [sg.Text('The above images could not be stitched. Please select an alternate method')],
            [sg.Button('Try Stitching With Other Images', key='-CHECKALLSTICHES-'), sg.Button('Skip over this image', key='-IGNORE-')]
        ]

        error_window = sg.Window(title='Stitch Error', layout=layout, finalize=True,
                    element_justification='c', return_keyboard_events=True)

        error_window['-ERROR-'].draw_image(data=fig2data(self.show_matches(im1_index, im2_index)), location=(18,75))

        while True:
            event, values = error_window.read()
            # print(event)
                
            if event == sg.WIN_CLOSED or event in ('Close', None):
                error_window.close()
                break

            if event == '-CHECKALLSTICHES-':
                for i in range(len(self.ims)):
                    if i != im2_index:
                        if self.check_matches(i, im2_index):
                            window[f'-BUTTON{im2_index+1}-'].update(button_color='green')
                            self.find_cpos(i, im2_index)
                            break
                error_window.close()
                break

            if event == '-IGNORE-':
                error_window.close()
                break

    def initial_stitch(self):
        '''
        creates and displays a stitched image of all selected images.
        '''
        for i in range(len(self.ims)):
            window[f'-BUTTON{i+1}-'].update(button_color='grey')
        window.refresh()
        self.normalise_im_sizes()
        self.ims[0].cxpos, self.ims[0].cypos = 0, 0
        window[f'-BUTTON1-'].update(button_color='green')
        time.sleep(0.1)
        window.refresh()
        for i in range(len(self.ims)-1):
            if self.check_matches(i, i+1) and self.ims[i].cxpos != None:
                window[f'-BUTTON{i+2}-'].update(button_color='green')
                self.find_cpos(i, i+1)
            else:
                window[f'-BUTTON{i+2}-'].update(button_color='red')
                self.backup_stitch(i, i+1)
            self.normalize_cpositions()
            stitch = self.draw_stitch()
            self.display_stitch(stitch)
            time.sleep(0.1)
            window.refresh()


'''Front End'''


sg.theme('DarkTeal12')

stitch_window = sg.Graph((1200,800), (0,0), (100,100), background_color='white', enable_events=True,
                            key='-STITCH-', drag_submits=True)

diffraction_window = sg.Image(size=(515,515), background_color='white', key='-DIFF-')

image_buttons = [sg.Button(button_text=str(i), expand_x=True, pad=(20,10), key=f'-BUTTON{i}-') for i in range(1, 21)]

layout = [
    [sg.Button('Update', key='-UPDATE-'), sg.Text('Folder'), sg.In(key='-FOLDER-', enable_events=True, expand_x=True, readonly=True), sg.FolderBrowse()],
    [sg.Frame(title='Loaded Images', layout=[image_buttons], expand_x=True, key='-LOADEDIMS-')],
    [sg.Frame(title='Stich Image', layout=[[stitch_window]], size=(850,850), element_justification='center', expand_x=True, expand_y=True),
        sg.Frame(title='Diffraction Image', layout=[[diffraction_window]], size=(555,555), element_justification='center')]  
]

window = sg.Window(title='Image Stitcher', layout=layout, finalize=True, resizable=True,
                    element_justification='c', size=(1920,1080), return_keyboard_events=True)

while True:  # Event Loop
    event, values = window.read()
    # print(event)

    if event == sg.WIN_CLOSED or event in ('Close', None):
        break

    if event == '-FOLDER-' or event == '-UPDATE-':
        stitch = Stitch(values['-FOLDER-'])
        stitch.initial_stitch()

    if 'stitch' in globals():

        if event == '-STITCH-':
            x,y = values['-STITCH-']
            if 'highlight' in globals():
                window['-STITCH-'].delete_figure(highlight)
            highlight = window['-STITCH-'].draw_circle((x,y), radius=3, line_color='red')
            stitch.display_diff(x,y)

        if event.startswith('-BUTTON'):
            i = int(event[7:-1])-1
            im = stitch.ims[i]
            top_left = (im.cxpos, im.cypos)
            bottom_right = (im.cxpos+im.pxsize, im.cypos+im.pysize)
            for _ in range(3):
                box = window['-STITCH-'].draw_rectangle(top_left, bottom_right, line_color='blue')
                window.refresh()
                time.sleep(0.1)
                window['-STITCH-'].delete_figure(box)
                window.refresh()
                time.sleep(0.1)