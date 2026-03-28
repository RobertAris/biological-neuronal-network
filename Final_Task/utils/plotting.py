import numpy as np
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

class Colour_Coding:
    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "gist_rainbow"):
        self.colourMap = cm.get_cmap(cmap)
        self.colourMap.set_bad('black')
        self.width = width
        self.electrodes = np.sort(np.unique(electrodesToCode))
        self.colours = np.zeros((len(self.electrodes),4))
    def __call__(self, electrodes):
        mask = np.in1d(electrodes,self.electrodes)
        colours = np.zeros((len(electrodes),4)) + np.nan
        # Use broadcasting to create an array of indices for each element in the first array
        colours[mask] = self.colours[np.searchsorted(self.electrodes, electrodes[mask])]
        return colours

class Circular_Angle_Colour_Coding(Colour_Coding):

    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "gist_rainbow"):
        super().__init__(electrodesToCode, width, cmap)

        coordsX = self.electrodes % self.width
        coordsY = self.electrodes // self.width
        centerX, centerY = (np.max(coordsX)+np.min(coordsX))/2, (np.max(coordsY)+np.min(coordsY))/2  # center of geometry

        # Compute angle in radians (-pi to pi)
        angles = np.arctan2(coordsY - centerY, coordsX - centerX)

        # Normalize angle to [0, 1] using the full theoretical range
        phi = (angles + np.pi) / (2 * np.pi)

        self.colours = self.colourMap(phi)

def create_post_stim_raster_plot(binned_spike_trains: np.ndarray, electrodes: np.ndarray):
    assert binned_spike_trains.ndim == 3, f"binned_spike_trains is expected to be a 3 dimensional tensor, got {binned_spike_trains.ndim }."
    assert len(electrodes) == binned_spike_trains.shape[1], f"#Electrodes ({len(electrodes)}) does not match #Channels ({binned_spike_trains.shape[1]})"

    colours = Circular_Angle_Colour_Coding(electrodes)(electrodes)
    weights = binned_spike_trains.sum(axis=1)
    p95 = np.percentile(weights, 95)
    weights = np.clip(weights / p95, 0, 1)
    image = np.einsum('scb,cd->sbd', binned_spike_trains, colours) # Adds colors at pixels
    image = np.clip(image / p95, 0, 1)
    image = image + 1*(1 - weights)[..., None] # Makes background white

    return np.clip(image, 0, 1)

def map_colour_to_electrode(
        axis: Axes,
        impedance_map: np.ndarray,
        electrodes: np.ndarray,
)-> (Axes):
    """
    Generates a plot of the voltage map, which is binarized into open/covered electrodes with 1/0 respectively.
    The colours are put ontop of the specified electrodes.
    :param axis: Axis of a matplotlib subplot.
    :param impedance_map: The voltage map, where covered electrodes have a value <= 0.
    :param electrodes: Electrodes, for which a blob should be plotted.
    :param colours: Colour of the respective blob.
    :return: axis, legend handels
    """
    colours = Circular_Angle_Colour_Coding(electrodes)(electrodes)
    normalization = Normalize(impedance_map.min(),impedance_map.max())
    voltageMapColourMap = plt.get_cmap("binary_r")
    coordsX = electrodes%impedance_map.shape[1]
    coordsY = electrodes//impedance_map.shape[1]
    bound = 10
    boundX = [max(int(np.min(coordsX)-bound),0),min(int(np.max(coordsX)+bound+1),impedance_map.shape[1])]
    boundY = [max(int(np.min(coordsY) - bound),0), min(int(np.max(coordsY) + bound+1),impedance_map.shape[0])]
    image = voltageMapColourMap(normalization(impedance_map[boundY[0]:boundY[1],boundX[0]:boundX[1]]))[:,:,:3]
    coordsX = coordsX - boundX[0]
    coordsY = coordsY - boundY[0]

    axis.imshow(image)
    axis.scatter(coordsX,coordsY,s=10,c=colours)
    axis.axis("off")
    return axis