import time
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def live_update_demo(blit = False):
    x = np.linspace(0,50., num=100)
    X,Y = np.meshgrid(x,x)
    fig = plt.figure()
    
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
    
    ax1 = plt.subplot( gs[0,0] ) #fig.add_subplot(2, 1, 1)
    ax2 = plt.subplot( gs[0:2,1:] ) #fig.add_subplot(2, 1, 2)
    ax3 = plt.subplot( gs[2,2] ) #fig.add_subplot(2, 2, 3)

    img = ax1.imshow(X, vmin=-1, vmax=1, interpolation="None", cmap="RdBu")


    line, = ax2.plot([], lw=3)
    text = ax2.text(0.8,0.5, "")

    ax2.set_xlim(x.min(), x.max())
    
    ax3.set_axis_off()
    text2 = ax3.text(0,0, "")
     

    fig.canvas.draw()   # note that the first draw comes before setting data 


    if blit:
        # cache the background
        axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
        ax2background = fig.canvas.copy_from_bbox(ax2.bbox)
        ax3background = fig.canvas.copy_from_bbox(ax3.bbox)

    plt.show(block=False)


    t_start = time.time()
    k=0.

    for i in np.arange(1000):
        img.set_data(np.sin(X/3.+k)*np.cos(Y/3.+k))
        line.set_data(x, np.sin(x/3.+k)+k/10)
        ax2.set_ylim([-1.1, k/10+1.1])
        
        text2.set_text(k)
        
        tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) ) 
        text.set_text(tx)
        #print tx
        k+=0.11
        if blit:
            # restore background
            fig.canvas.restore_region(axbackground)
            fig.canvas.restore_region(ax2background)
            fig.canvas.restore_region(ax3background)

            # redraw just the points
            ax1.draw_artist(img)
            ax2.draw_artist(line)
            ax2.draw_artist(text)
            ax3.draw_artist(text2)

            # fill in the axes rectangle
            fig.canvas.blit(ax1.bbox)
            #fig.canvas.blit(ax2.bbox)
            fig.canvas.blit(ax3.bbox)

            # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # it is mentionned that blit causes strong memory leakage. 
            # however, I did not observe that.

        else:
            # redraw everything
            fig.canvas.draw()

        fig.canvas.flush_events()
        #alternatively you could use
        #plt.pause(0.000000000001) 
        # however plt.pause calls canvas.draw(), as can be read here:
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html


live_update_demo(False)   # 175 fps
#live_update_demo(False) # 28 fps
