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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import PIL
import copy
import re #Regular Expressions module

def fig2data(fig: np.ndarray) -> PIL.Image:
    '''coverts a numpy array into a PIL image so that it can be displayed in PySimpleGUI'''
    img = PIL.Image.fromarray(fig)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()

def timer(func): #Timer Wrapper
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        run_time = end - start
        time_string = f'{run_time:.3f} s' if run_time >= 1 else f'{run_time*1E3:.3f} µs' if run_time >= 0.001 else f'{run_time*1E6:.3f} ns'
        print(f'{func} method took {time_string} to run')
    return wrapper

'''Back End'''

class Micrograph():
    '''Stores key micrograph parameters as attributes'''

    def __init__(self, file_list: list[str], id: str) -> None: #check all files for each bit of data raising exception for data not found
        self.file_list = file_list
        self.id = id
        self.jpgpath = [file for file in file_list if id in file and '.jpg' in file][0]
        self.metadatapath = [file for file in file_list if id in file and '.hdf' in file and '.hdf5' not in file][0]
        self.imdatapath = [file for file in file_list if id in file and '.hdf5' in file][0]
        
        self.imarray = cv2.imread(self.jpgpath, cv2.IMREAD_GRAYSCALE)
        self.pysize, self.pxsize = np.shape(self.imarray) #size of each axis in pixels
        self.scale_factor = 1 #origional size * scale_factor = current size
        self.cxpos = self.cypos = None #computed x an y positions of each image
        
        hdf = h5.File(self.metadatapath, 'r')
        self.pixel_metre_length = np.array(hdf['metadata/step_size(m)'])
        self.mag = np.array(hdf['metadata/magnification']) # true size * mag = observed size
        hdf.close()

        sift = cv2.SIFT_create()
        self.features, self.descriptors = sift.detectAndCompute(self.imarray, None)

        hdf = h5.File(self.imdatapath, 'r')
        #loading diffraction data into storage
        hdf_data = da.from_array(hdf['Experiments/__unnamed__/data'], chunks='auto')
        #creating arrays from data that are only loaded into memory as needed
        self.difdata = hs.signals.Signal2D(hdf_data).as_lazy().data

class Stitch():
    '''Stitches together micrograph images to form a stitch image array'''
    
    def __init__(self):
        self.stitch_mag = None
        self.stitch_pixel_metre_length = None
        self.data = np.empty(0)
        self.ims = []

    def normalise_im_mags(self) -> None:
        '''
        scales all image sizes to highest mag size using cubic interpolation.
        This means images are only scaled up so resolution of each im is at least 1 pixel of the origional im
        '''
        self.stitch_mag = max([im.mag for im in self.ims])
        self.stitch_pixel_metre_length = min([im.pixel_metre_length for im in self.ims])
        for im in self.ims:
            if im.mag != self.stitch_mag:
                im.scale_factor = self.stitch_mag/im.mag
                im.pysize, im.pxsize = round(im.pysize*im.scale_factor), round(im.pxsize*im.scale_factor)
                im.imarray = cv2.resize(im.imarray, (im.pxsize, im.pysize), interpolation=cv2.INTER_CUBIC)
                im.mag = copy.copy(self.stitch_mag)
    
    def best_matches(self, im1_index: int, im2_index: int) -> list[cv2.DMatch]:
        '''returns the 5 best matches for any two micrograph ims in descending order'''
        im1, im2 = self.ims[im1_index], self.ims[im2_index]
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(im1.descriptors, im2.descriptors)
        return sorted(matches, key=lambda x:x.distance)[:5]

    def best_offsets(self, im1_index: int, im2_index: int) -> list[list[float, float]]:
        '''returns the 5 best offsets [(xoff, yoff)] for the im2 relative to im 1'''
        best_matches = self.best_matches(im1_index, im2_index)
        im1, im2 = self.ims[im1_index], self.ims[im2_index]
        offsets = []
        for match in best_matches:
            feature1, feature2 = im1.features[match.queryIdx], im2.features[match.trainIdx]
            xoff = feature1.pt[0]*im1.scale_factor-feature2.pt[0]*im2.scale_factor
            yoff = feature1.pt[1]*im1.scale_factor-feature2.pt[1]*im2.scale_factor
            offsets.append((xoff, yoff))
        return np.array(offsets)

    def check_matches(self, im1_index: int, im2_index: int) -> bool:
        '''returns True if the top matches for any two micrograph ims consistently return the same offset. Otherwise returns False'''
        best_offsets = self.best_offsets(im1_index,im2_index)
        tolerance = max([im.scale_factor for im in self.ims])
        return np.std(best_offsets[:,0]) < tolerance and np.std(best_offsets[:,1]) < tolerance

    def find_cpos(self, im1_index: int, im2_index: int) -> None:
        '''finds the calculated offset of the second image compared to the first and uses this to find the calculated position'''
        xoff, yoff = np.round(np.mean(self.best_offsets(im1_index, im2_index), axis=0))
        self.ims[im2_index].cxpos, self.ims[im2_index].cypos = self.ims[im1_index].cxpos + int(xoff), self.ims[im1_index].cypos + int(yoff)

    def normalize_cpositions(self) -> None:
        '''normalises the calculated positions of images so the bottom left is at 0,0'''
        miny, minx = min([im.cypos for im in self.ims]), min([im.cxpos for im in self.ims])
        for im in self.ims:
            im.cypos -= miny
            im.cxpos -= minx

    def draw_blurred_stitch(self) -> np.ndarray: #not used as it comes out looking funny
        '''returns an array of the stitched image with overlapping images blurred together'''
        size = [max([im.cypos+im.pysize for im in self.ims]), max([im.cxpos+im.pxsize for im in self.ims])]
        canvas = np.full(size, 0, dtype=int)
        coverage = np.full(size, 0, dtype=int)
        figure = np.full(size, 255, dtype=np.uint8)
        for im in self.ims:
            canvas[im.cypos:im.cypos+im.pysize, im.cxpos:im.cxpos+im.pxsize] += im.imarray
            coverage[im.cypos:im.cypos+im.pysize, im.cxpos:im.cxpos+im.pxsize] += 1
        figure[(coverage!=0)] = canvas[(coverage!=0)] / coverage[(coverage!=0)]
        return figure

    def find_diff(self, x: int, y: int) -> dict[str: np.ndarray] | bool:
        '''
        returns a dictionary with the key being the image id and the value being the image array

        if no diffraction image is found returns False

        For scaled images the diffraction pattern of the nearest pixel in the unscaled image is used
        '''
        dif_ims = dict()
        for im in self.ims:
            if im.cypos <= y < im.cypos+im.pysize and im.cxpos <= x < im.cxpos+im.pxsize:
                dify, difx = round((y-im.cypos)/im.scale_factor), round((x-im.cxpos)/im.scale_factor)
                dif_ims[im.id] = im.difdata[dify,difx].compute()
        return dif_ims if len(dif_ims) > 0 else False
    
    def try_add_im(self, new_im: Micrograph) -> bool:
        '''
        Attemps to add micrograph im to stitch. Returns True if this succeeds else False.
        '''
        self.ims.append(new_im)
    
        if len(self.ims) == 1:
            self.ims[0].cxpos, self.ims[0].cypos = 0, 0
            self.stitch_mag = copy.copy(new_im.mag)
            self.data = np.copy(new_im.imarray)
            new_im.cxpos, new_im.cypos = 0, 0
            return True
        
        self.normalise_im_mags()
        for im_index in range(len(self.ims[:-1])):
            if self.check_matches(im_index, -1):
                self.find_cpos(im_index, -1)
                self.normalize_cpositions()
                return True
        
        del self.ims[-1]
        return False

    def find_im_coords(self, im_id: str) -> list[list[int,int], list[int,int]] | bool:
        '''
        returns the top left and bottom right x,y coordinates of image with the specified image id
        if the image with the specified image id is not found, return False
        '''
        for im in self.ims:
            if im.id == im_id:
                return [im.cxpos, im.cypos], [im.cxpos+im.pxsize, im.cypos+im.pysize]
        return False

class App():
    '''Creates Micrograph and Stitch objects and returns images to be used in the interface'''

    @timer
    def __init__(self, folder_path: str) -> None | str:
        '''Loads in data and creates stitches. Returns error string if process fails'''
        self.files = self.find_files(folder_path)
        self.ids = self.find_ids(self.files)
        response = self.check_data(self.files, self.ids)
        if response is not True:
            print(f'ERROR: {response}')
            return response
        self.ims = [Micrograph(self.files, id) for id in self.ids]
        self.stitches = []

        while len(self.ims) > 0:
            self.stitches.append(Stitch())
            self.stitches[-1].try_add_im(self.ims[0])
            del self.ims[0]
            self.try_stitching_all_ims()

        for stitch in self.stitches:
            stitch.data = stitch.draw_blurred_stitch()
    
    def find_files(self, folder_path: str) -> list[str]:
        '''returns all files in the given folder path'''
        file_list = []
        for path, folders, files in os.walk(folder_path):
            for file in files:
                file_list.append(os.path.join(path, file))
        return sorted(file_list)

    def find_ids(self, file_list: list[str]) -> list[str]:
        '''returns all date time ids (YYYYMMDD_hhmmss) in given file list'''
        ids = set()
        pattern = r'[\d]{8}\_[\d]{6}' #regex pattern of 8 numbers then _ then 6 more numbers
        for file in file_list:
            for match in re.findall(pattern, file):
                ids.add(match)
        return sorted(list(ids))

    def check_data(self, file_list: list[str], ids: list[str]) -> bool | str:
        '''
        checks to make sure that all neccessary data is present in file_list for each id
        returns true if all data is present. Otherwise returns string describing what data is missing
        '''
        for id in ids:
            try:
                jpg_path = [file for file in file_list if id in file and '.jpg' in file][0]
                _ = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            except:
                return f'complete image jpg data could not be found for image {id}'
            
            try:
                metadata_path = [file for file in file_list if id in file and '.hdf' in file and '.hdf5' not in file][0]
                hdf = h5.File(metadata_path, 'r')
                _ = int(np.array(hdf['metadata/magnification']))
                hdf.close()
            except:
                return f'complete image metadata coulf not be found for image {id}. A .hdf file with an integer at metadata/magnification expected'
            
            try:
                imdata_path = [file for file in file_list if id in file and '.hdf5' in file][0]
                hdf = h5.File(imdata_path, 'r')
                _ = da.from_array(hdf['Experiments/__unnamed__/data'], chunks='auto')
                hdf.close()
            except:
                return f'complete image diffraction data could not be found for image {id}. A .hdf5 with a numpy array at Experiments/__unnamed__/data parameter expected'
            
            return True

    def try_stitching_all_ims(self) -> None:
        '''Attempt to stitch all remaining images to existing stitches'''  
        ims_added_to_stitch = True
        while ims_added_to_stitch is True:
            ims_added_to_stitch = False
            for im_index, im in enumerate(self.ims):
                for stitch in self.stitches:
                    if stitch.try_add_im(im):
                        ims_added_to_stitch = True
                        del self.ims[im_index]
                        break
    
    def find_stitch_image(self, stitch_index: int) -> np.ndarray:
        '''returns a numpy array image of the stitched together micrographs for a given stitch'''
        return self.stitches[stitch_index-1].data

    def find_stitch_pixel_metre_length(self, stitch_index: int) -> float:
        '''returns the length in metres of each pixel for a given stitch'''
        return self.stitches[stitch_index-1].stitch_pixel_metre_length

    def find_diff_images(self, stitch_index: int, x: int, y: int) -> dict[str: np.ndarray] | bool:
        '''
        returns a dictionary with the key being the image id and the value being the image array

        if no diffraction image is found returns False

        For scaled images the diffraction pattern of the nearest pixel in the unscaled image is used
        '''
        return self.stitches[stitch_index-1].find_diff(x=x, y=y)
    
    def find_im_coords(self, im_id: str) -> list[list[int,int], list[int,int], int]:
        '''returns the top left and bottom right x,y coordinates of the image with the specified image id as well as the stitch the image is in'''
        for stitch_index in range(len(self.stitches)):
            response = self.stitches[stitch_index].find_im_coords(im_id)
            if response != False:
                return [response[0], response[1], stitch_index]

'''Front End'''

sg.theme('DarkTeal12')
          
def stitch_layout(stitch_number: int) -> list[list[sg.Column]]:
    return [[sg.Column(key=f'-SCOL{stitch_number}-', scrollable=True, expand_x=True, expand_y=True, element_justification='center',
                layout = [[sg.Graph(canvas_size=(1000,1000), graph_bottom_left=(0,0), graph_top_right=(100,100), background_color='white',
                           enable_events=True, key=f'-STITCH{stitch_number}-', drag_submits=True, expand_x=True, expand_y=True)]])
                ]]

stitch_tabs = [
    [sg.Tab(title='Stitch 1', layout=stitch_layout(1), key='-STAB1-', element_justification='center', expand_x=True, expand_y=True)],
    [sg.Tab(title=f'Stitch {i}', layout=stitch_layout(i), key=f'-STAB{i}-', element_justification='center', visible=False, expand_x=True, expand_y=True) for i in range(2,21)]
    ]

stitch_tab_group = sg.TabGroup(layout=stitch_tabs, enable_events=True, key='-STABGROUP-', pad=(10,10), expand_x=True, expand_y=True)

zoom_slider = sg.Slider((0.5, 5), default_value=1.0, resolution=0.5, orientation='vertical', key='-ZOOM-', enable_events=True, expand_y=True, 
            disable_number_display=True, pad=(10, 30), tick_interval=0.5)

zoom_column = sg.Column(layout=[[sg.Text('Zoom')], [zoom_slider]], element_justification='center', expand_y=True)

stitch_frame = sg.Frame(title='Stitch Images',
                        layout=[[zoom_column, stitch_tab_group]],
                        element_justification='center', expand_x=True, expand_y=True)

def diffraction_layout(diffraction_number: int) -> list[list[sg.Image]]:
    return [
        [sg.Text('Micrograph id'), sg.Text('YYYYMMDD_hhmmss', key=f'-DIFFIDSTRING{diffraction_number}-', enable_events=True)],
        [sg.Image(size=(515,515), background_color='white', key=f'-DIFF{diffraction_number}-')]]

diffraction_tabs = [
    [sg.Tab(title='Image 1', layout=diffraction_layout(1), key='-DIFFTAB1-')],
    [sg.Tab(title=f'Image {i}', layout=diffraction_layout(i), key=f'-DIFFTAB{i}-', visible=False) for i in range(2,10)]
]

diffraction_tab_group = sg.TabGroup(layout=diffraction_tabs, enable_events=True, key='-DIFFTABGROUP-', pad=(10,10))

threshold_slider = [sg.Text('Threshold', pad=(5, 10)),
                    sg.Slider(range=(0,100), orientation='horizontal', disable_number_display=True, size=(30,20), enable_events=True, key='-TSLIDER-'),
                    sg.Text('0', size=(3,1), enable_events=True, key='-TTEXT-')]


diffraction_frame = sg.Frame(title='Diffraction Images', element_justification='center', layout=[
                                    [sg.Text('Stitch Position'), sg.Text('x:0 y:0', key='-SPOSTEXT-', enable_events=True)],
                                    [diffraction_tab_group],
                                    threshold_slider])

scalebar_locs = ['upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center', 'lower right']

stitch_scalebar_frame = sg.Frame('Stitch Scale Bar', expand_x=True, expand_y=True, element_justification='left', layout=[
    [sg.Text('Metre Length', size=(16,1)), sg.Input(default_text=100, key='-SBLEN-', enable_events=True, size=(6,1)),
        sg.Spin(values=['nm', 'µm', 'mm'], initial_value='nm', key='-SBUNIT-', enable_events=True, size=(4,1))],
    [sg.Text('Line Thickness', size=(16,1)), sg.Spin(values=list(range(1, 50)), initial_value=1, key='-SBTHICKNESS-', enable_events=True, size=(4,1))],
    [sg.Text('Label Location', size=(16,1)), sg.Combo(scalebar_locs, default_value='lower right', key='-SBLOC-', enable_events=True)],
    [sg.Text('Font Size', size=(16,1)), sg.Spin(values=list(range(1, 100)), initial_value=10, key='-SBFONTSIZE-', enable_events=True, size=(4,1))],
    [sg.Push(), sg.Text('Analysis View'), sg.Slider(range=(0,1), default_value=0, orientation='horizontal', disable_number_display=True, size=(7,20), key='-SBTOGGLE-', enable_events=True, pad=15), sg.Text('Scale Bar View'), sg.Push()]
])

my_plug = sg.Text('Check out my GitHub at /AyhamSaffar for more apps like this', text_color='grey')

image_buttons = [sg.Button(button_text=str(i), expand_x=True, pad=(10,10), key=f'-BUTTON{i}-') for i in range(1, 31)]

export_button = sg.FileSaveAs('Export Current Stitch', target='-EXPORTPATH-', file_types=(('PNG', '.png'),), key='-EXPORT-', enable_events=True, disabled=True)
hidden_input = sg.Input(enable_events=True, key='-EXPORTPATH-', visible=False)

layout = [
    [export_button, hidden_input, sg.Text('Folder'), sg.In(key='-FOLDER-', enable_events=True, expand_x=True, readonly=True), sg.FolderBrowse()],
    [sg.Frame(title='Loaded Images', layout=[image_buttons], expand_x=True, key='-LOADEDIMS-')],
    [stitch_frame, sg.Column(layout=[[diffraction_frame], [stitch_scalebar_frame], [my_plug]], element_justification='center')] 
]

window = sg.Window(title='Image Stitcher', layout=layout, finalize=True, resizable=True,
                    element_justification='center', size=(1920,1080), return_keyboard_events=True)

#fixing the layout manually so it fills all the space
info = zoom_column.Widget.pack_info()
info.update({'expand':0})
zoom_column.Widget.pack(**info, after=stitch_tab_group.Widget)

#front end helper functions

def general_popup(title: str, text: str):
    '''Creates a blocking popup window with the given title and text'''
    layout = [[ sg.Text(text)], [sg.Column([[sg.Button('Close', key='-CLOSE-', enable_events=True)]], justification='center') ]]
    window = sg.Window(title=title, layout=layout)
    while True:  # Event Loop
        event, values = window.read()
        print(event)
        if event == sg.WIN_CLOSED or event in ('-CLOSE-', None):
            window.close()
            break

def add_scale_bar(im: np.ndarray, stitch_index:int, length: int, length_unit: str, thickness: int, position: str, font: int) -> np.ndarray:
    '''
    Adds a scale bar to the given stitch array using MatPlotLib.
    Requires the true metre length of the scale bar as well as its pixel thickness and position.
    '''
    scalebar_true_length = length / (1E3 if length_unit=='mm' else 1E6 if length_unit=='µm' else 1E9)
    scalebar_pixel_length = round(scalebar_true_length / app.find_stitch_pixel_metre_length(stitch_index))

    fig = plt.figure(frameon=False)
    ax = plt.axes((0,0,1,1))
    ax.axis("off")
    ax.matshow(im, cmap='gray', interpolation='none')
    scalebar = AnchoredSizeBar(transform=ax.transData, size=scalebar_pixel_length, label=f'{length} {length_unit}', loc=position,
                                pad=0.5, color='black', frameon=True, size_vertical=thickness, fontproperties={'size': font})
    ax.add_artist(scalebar)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 3)
    data = np.dot(data, [0.2989, 0.5870, 0.1140]) #convert RGB back to grey
    plt.close()
    return np.array(np.round(data), dtype=np.uint8)

def display_stitch(stitch_index: int, zoom: float, include_scalebar: bool) -> np.ndarray:
    '''displays an image of the stitched together microgrographs for the given stitch. Returns a numpy array of the displayed stitch'''
    im_array = app.find_stitch_image(stitch_index)
    if include_scalebar == True:
        im_array = add_scale_bar(im=im_array, stitch_index=stitch_index, length=int(values['-SBLEN-']), length_unit=values['-SBUNIT-'], thickness=values['-SBTHICKNESS-'], position=values['-SBLOC-'], font=values['-SBFONTSIZE-'])
    im_p_height, im_p_width = np.shape(im_array)
    im_height, im_width = int(im_p_height*zoom), int(im_p_width*zoom)
    im_array = cv2.resize(im_array, (im_width, im_height), interpolation=cv2.INTER_CUBIC if include_scalebar==True else cv2.INTER_NEAREST)
    window[f'-STITCH{stitch_index}-'].set_size(size=(im_width, im_height))
    window[f'-STITCH{stitch_index}-'].change_coordinates((0,im_p_height), (im_p_width, 0))
    window[f'-STITCH{stitch_index}-'].draw_image(data=fig2data(im_array), location=(0,0))
    window.refresh()
    window[f'-SCOL{stitch_index}-'].contents_changed()
    return im_array

def display_diff(stitch_index: int, x: int, y: int, threshold: int) -> None | bool:
    '''
    Displays diffraction image at (x,y) of the selected stitch image in the diffraction box with the set threshold
    Returns False if no diffraction data is present at the (x,y) coordinates
    '''
    for i in range(2, 10):
       window[f'-DIFFTAB{i}-'].update(visible=False) 
    diff_data = app.find_diff_images(stitch_index=stitch_index, x=x, y=y)
    if diff_data is False:
        return False
    for i, [id, fig] in enumerate(diff_data.items()):
        window[f'-DIFFTAB{i+1}-'].update(visible=True)
        window[f'-DIFFIDSTRING{i+1}-'].update(id)
        fig[fig<threshold] = 0
        fig = fig * (255 // np.max(fig))
        window[f'-DIFF{i+1}-'].update(fig2data(fig))

def log_diff_tab_im_ids(stitch_index: int, x: int, y: int) -> dict[str: str]:
    '''creates a dictionary of {diffraction tab keys: image ids}'''
    tab_dict = dict()
    diff_data = app.find_diff_images(stitch_index=stitch_index, x=x, y=y)
    for i, [id, fig] in enumerate(diff_data.items()):
        tab_dict[f'-DIFFTAB{i+1}-'] = id
    return tab_dict

def flash_image(im_id: str) -> None:
    '''Flashes a blue border around the selected image in the stitch window'''
    top_left, bottom_right, stitch_index = app.find_im_coords(im_id=im_id)
    window[f'-STAB{stitch_index+1}-'].select()
    for _ in range(3):
        box = window[f'-STITCH{stitch_index+1}-'].draw_rectangle(top_left, bottom_right, line_color='blue')
        window.refresh()
        time.sleep(0.1)
        window[f'-STITCH{stitch_index+1}-'].delete_figure(box)
        window.refresh()
        time.sleep(0.1)

while True:  # Event Loop
    event, values = window.read()
    # print(event, values)

    if event == sg.WIN_CLOSED or event in ('Close', None):
        break
    
    if event == '-SBLEN-':
        if values['-SBLEN-'] == '':
            window['-SBLEN-'].update(value='1')
            values['-SBLEN-'] = '1'
        try:
            int(values['-SBLEN-'])
        except:
            input = values['-SBLEN-']
            general_popup(title='Bad Input', text=f'The entry {input} is not a valid scalebar length')
            window['-SBLEN-'].update(value=values['-SBLEN-'][:-1])
            values['-SBLEN-'] = values['-SBLEN-'][:-1]

    if event == '-FOLDER-':
        app = App(values['-FOLDER-'])
        if isinstance(app, str):
            general_popup(title='Error', text=app)
            continue
        for i in range(1, len(app.stitches)+1):
            window[f'-STAB{i}-'].update(visible=True)
        stitch = display_stitch(stitch_index=1, zoom=values['-ZOOM-'], include_scalebar=False)
        for i, id in enumerate(app.ids):
            window[f'-BUTTON{i+1}-'].update(text=id)
        for i in range(len(app.ids)+1, 31):
            window[f'-BUTTON{i}-'].update(visible=False)
        window['-EXPORT-'].update(button_color='green', disabled=False)

    if 'stitch' in globals():

        if event in ['-ZOOM-', '-STABGROUP-']:
            stitch = display_stitch(stitch_index=int(values['-STABGROUP-'][-2]), zoom=values['-ZOOM-'], include_scalebar=values['-SBTOGGLE-'])

        if event == '-SBTOGGLE-':
            if 'highlight' in globals():
                window[highlight['element_key']].delete_figure(highlight['figure'])
            stitch = display_stitch(stitch_index=int(values['-STABGROUP-'][-2]), zoom=values['-ZOOM-'], include_scalebar=values['-SBTOGGLE-'])

        if event == '-EXPORTPATH-':
            stitch_tab = int(values['-STABGROUP-'][-2])
            if values['-SBTOGGLE-'] == 0:
                with PIL.Image.fromarray(app.find_stitch_image(stitch_tab)) as export_im:
                    export_im.save(values['-EXPORTPATH-'])
                general_popup(title='Stitch Saved Successfully', 
                    text=f'An unaltered image of stitch {stitch_tab} was successfully saved \r\rIn order to save this stitch as a scaled smoothed image with a scalebar, select Scale Bar View before exporting')
            if values['-SBTOGGLE-'] == 1:
                with PIL.Image.fromarray(stitch) as export_im:
                    export_im.save(values['-EXPORTPATH-'])
                general_popup(title='Stitch Saved Successfully', 
                    text=f'A scaled smoothed image of stitch {stitch_tab} with a scalebar was successfully saved \r\rIn order to save this stitch as an unaltered image, select Analysis View before exporting')

        if values['-SBTOGGLE-'] == 0:
            if 'STITCH' in event and '+UP' not in event: #one of the stitch images are clicked
                x, y = values[event]
                diff = display_diff(stitch_index=int(values['-STABGROUP-'][-2]), x=x, y=y, threshold=int(values['-TSLIDER-']))
                if diff != False:
                    window['-SPOSTEXT-'].update(f'x:{x} y:{y}')
                    if 'highlight' in globals():
                        window[highlight['element_key']].delete_figure(highlight['figure'])
                    highlight = {'element_key': event, 'figure': window[event].draw_rectangle((x-1, y-1), (x+1, y+1), line_color='red')}
                    window['-DIFFTAB1-'].set_focus(force=True)
                    tab_dict = log_diff_tab_im_ids(stitch_index=int(values['-STABGROUP-'][-2]), x=x, y=y)
            
            if event.startswith('-BUTTON'):
                button_number = int(event[7:-1])
                id = app.ids[button_number-1]
                flash_image(im_id=id)

        if values['-SBTOGGLE-'] == 1:
            if event in ['-SBTOGGLE-', '-SBLEN-', '-SBUNIT-', '-SBTHICKNESS-', '-SBLOC-', '-SBFONTSIZE-']:
                stitch = display_stitch(stitch_index=int(values['-STABGROUP-'][-2]), zoom=values['-ZOOM-'], include_scalebar=True)
    
        if 'diff' in globals():

            if event == '-TSLIDER-':
                window['-TTEXT-'].update(int(values['-TSLIDER-']))
                diff = display_diff(stitch_index=int(values['-STABGROUP-'][-2]), x=x, y=y, threshold=int(values['-TSLIDER-']))

            if event == '-DIFFTABGROUP-' and values['-SBTOGGLE-'] == 0:
                im_id = tab_dict[values[event]]
                flash_image(im_id=im_id)

window.close()
