import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

COLOR = {
    'Incision': 'red',
    'Stitch': 'green',
    'incision': 'red',
    'stitch': 'green'
}

def visualize(data, image_id):
    image_id = str(image_id)
    image_path = Path('data/images/default')/data[image_id]['file']
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # plot incision
    points = data[image_id]['incision']
    x_coords, y_coords = zip(*points)
    x_coords = np.array(x_coords, dtype=float)
    y_coords = np.array(y_coords, dtype=float)
    ax.plot(x_coords, y_coords, color='g', linewidth=2)
    
    # plot incision polyline points
    for i in range(len(x_coords)):
        plt.plot(x_coords[i], y_coords[i],'xc')
    
    # plot all the stitches
    for i in range(len(data[image_id]['stitches'])):
        points = data[image_id]['stitches'][i]
        x_coords, y_coords = zip(*points)
        x_coords = np.array(x_coords, dtype=float)
        y_coords = np.array(y_coords, dtype=float)
        ax.plot(x_coords, y_coords, color='r', linewidth=2)

def visualize_old(data, image_id):
    '''Visualizes image and it's Incission and Stitch annotations'''

    image_path = Path('data/images/default')/data[image_id]['file']
    image = mpimg.imread(image_path)
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    for idx, label in enumerate(data[image_id]['label']):
        points = data[image_id]['points'][idx]
        x_coords, y_coords = zip(*points)
        x_coords = np.array(x_coords, dtype=float)
        y_coords = np.array(y_coords, dtype=float)
        ax.plot(x_coords, y_coords, color=COLOR[label], linewidth=2)
    # Show the plot
    plt.show()
