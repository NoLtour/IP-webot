from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
from time import sleep as delay
from time import time as getTime
import numpy as np

from testing import CartesianPose
 

FIGURE_SCALING = 1

class PlotWindow:
    """ Acts as a window for hosting displays, it's broken into a grid onto which displays are placed """
    
    def __init__(this, width=3, height=3, maxRPS=15): 
        this.gridSP = gridspec.GridSpec(width, height )
        
        this.width = width
        this.height = height
        this.minRenderDelay = 1/maxRPS
        
        this.fig = plt.figure( figsize=(height*FIGURE_SCALING, width*FIGURE_SCALING) )
        
        this.slaveDisplays = []
        
        this.rendered = False
        this.lUpdate = getTime()
        
    def render(this):
        """ Call to render changes """
        
        cTime = getTime()
        if ( cTime - this.lUpdate < this.minRenderDelay ):
            return
        
        this.lUpdate = cTime
        
        if ( this.rendered ):
            madeChanges = False
            
            for cDisplay in this.slaveDisplays:  
                if ( cDisplay.changed ):
                    madeChanges = True
                    
                    this.fig.canvas.restore_region( cDisplay.cachedBackground )
                    cDisplay.update() 
                    this.fig.canvas.blit( cDisplay.subAxis.bbox )
                    
            if ( madeChanges ):
                this.fig.canvas.flush_events()
                     
        else:
            this.fig.canvas.draw()
            
            for cDisplay in this.slaveDisplays:  
                cDisplay.cachedBackground = this.fig.canvas.copy_from_bbox( cDisplay.subAxis.bbox )
                
            plt.show(block=False)
            
            this.rendered = True
    
class PlotDisplay:
    """ Contains code related to the subfigure, links to a "PlotWindow". Generic class for inheritance """
    
    def __init__(this, xPosition, yPosition, width, height, parentWindow ):
        this.width  = width
        this.height = height
        this.xPosition = xPosition
        this.yPosition = yPosition
        
        if ( width < 0 or height < 0 ):
            raise ValueError( "display has invalid dimensions" ) 
        # TODO add more dimension validation later
                
        this.parentWindow = parentWindow
        
        this.subAxis = plt.subplot( parentWindow.gridSP[yPosition:yPosition+height, xPosition:xPosition+width ] )
        
        this.cachedBackground = None
        this.changed = True
        
        parentWindow.slaveDisplays.append( this )
        
        pass
    
    def update(this):
        """ Re renders this plot using the stored data """
        
        this.changed = False 
    
    def parseData(this):
        """ Parses data for storage """
        
        this.changed = True 
    
class ValueDisplay( PlotDisplay ):
    """Displays a single value, can be any stringifiable datatype"""
    
    def __init__(this, xPosition, yPosition, width, parentWindow, name):
        
        super().__init__( xPosition, yPosition, width, 1, parentWindow )
        this.textElement = this.subAxis.text(0,0,"")
        this.subAxis.set_axis_off()
        
        this.name = name
        this.value = "unset"
        
    def parseData(this, newValue):
        super().parseData()
        this.value = str( newValue ) 
        
    def update(this):
        super().update()
        this.textElement.set_text( this.name+": "+this.value )
        this.subAxis.draw_artist( this.textElement )
    
class LineGraphDisplay( PlotDisplay ):
    """Simple x-y plot, uses fixed axis"""
    
    def __init__(this, xPosition, yPosition, width, height, parentWindow, xMin, xMax, yMin, yMax, xLabel="", yLabel="", title=""): 
        super().__init__( xPosition, yPosition, width, height, parentWindow ) 
        
        this.graph, = this.subAxis.plot([], lw=3)
        this.subAxis.set_xlim( xMin, xMax )
        this.subAxis.set_ylim( yMin, yMax )
        
        this.xValues = []
        this.yValues = []
        
    def parseData(this, xValues, yValues):
        super().parseData() 
        this.xValues = xValues
        this.yValues = yValues
        
    def update(this):
        super().update()
        this.graph.set_data( this.xValues, this.yValues )
        this.subAxis.draw_artist( this.graph )
        
    
class GidGraphDisplay( PlotDisplay ):
    """Simple x-y plot, uses fixed axis"""
    
    def __init__(this, xPosition, yPosition, width, height, parentWindow,  xMax,   yMax, xLabel="", yLabel="", title=""): 
        super().__init__( xPosition, yPosition, width, height, parentWindow )  
         
        this.gData = np.random.random( (xMax, yMax) )
        
        this.xMax = xMax
        this.yMax = yMax
        
        this.gridGraph = this.subAxis.imshow(this.gData, cmap='gray', interpolation='none', origin='lower', extent=[0, xMax, 0, yMax])
        
        this.subAxis.set_xlim( 0, xMax )
        this.subAxis.set_ylim( 0, yMax )
        
    def randomData(this): 
        this.parseData( np.random.random( (this.width, this.height) ) )
        
    def parseData(this, nData):
        super().parseData() 
        this.gData = nData
        
    def update(this):
        super().update()
        this.gridGraph.set_data( this.gData )
        this.subAxis.draw_artist( this.gridGraph )
        


class PoseDisplay( PlotDisplay ):
    """Displays poses from a topdown perspective"""
    
    def __init__(this, xPosition, yPosition, width, height, parentWindow, xMin, xMax, yMin, yMax, xLabel="", yLabel="", title=""): 
        super().__init__( xPosition, yPosition, width, height, parentWindow ) 
         
        this.vecGraph = this.subAxis.quiver([0],[0],[0],[0])
        this.subAxis.set_xlim( xMin, xMax )
        this.subAxis.set_ylim( yMin, yMax )
        
        this.poses = [ CartesianPose( 0, 0, 0, 0, 0, 0 ) ]
        
        this.xPos = []
        this.yPos = []
        this.uCmp  = []
        this.vCmp  = []
        
    def parseData(this, newPoseList):
        super().parseData() 
        this.poses = newPoseList
        
        this.xPos = []
        this.yPos = []
        this.uCmp  = []
        this.vCmp  = []
        
        for targetPose in newPoseList:
            this.xPos.append( targetPose.x)
            this.yPos.append( targetPose.y)
            this.uCmp.append( np.cos(targetPose.yaw) )
            this.vCmp.append( np.sin(targetPose.yaw) )
        
    def update(this):
        super().update()
        
        this.vecGraph.set_UVC( this.uCmp, this.vCmp )  
        this.vecGraph.set_offsets( np.array([[this.xPos[0], this.yPos[0]]]) )
        this.subAxis.draw_artist( this.vecGraph )

Arrow_Length = 0.4
Arrow_HWidth = 0.2

class RobotDisplay( PlotDisplay ):
    """Displays poses from a topdown perspective"""
    
    def __init__(this, xPosition, yPosition, width, height, parentWindow, xMin, xMax, yMin, yMax, xLabel="", yLabel="", title=""): 
        super().__init__( xPosition, yPosition, width, height, parentWindow ) 
        
        this.graphRBody, = this.subAxis.plot([], lw=3)
        
        this.subAxis.grid()
        this.subAxis.set_xlim( xMin, xMax )
        this.subAxis.set_ylim( yMin, yMax )
        
        this.graphPCloud, = this.subAxis.plot([], lw=3) 
        
        this.robotPosRend = [[0,0,0],[0,0,0]]
        this.pointCloud   = [[0],[0]]
        
    def parseData(this, robotPose, pointCloud):
        super().parseData() 
        
        this.getRobotPosRender( robotPose )
        this.pointCloud   = pointCloud
    
    def getRobotPosRender(this, robotPose):
        rX = robotPose.x
        rY = robotPose.y
        rA = robotPose.yaw + np.pi
        
        this.robotPosRend = [
            [ rX + ( Arrow_Length*np.cos(rA) + Arrow_HWidth*np.sin(rA) ), rX, rX + ( Arrow_Length*np.cos(rA) - Arrow_HWidth*np.sin(rA) ) ],
            [ rY + ( Arrow_Length*np.sin(rA) - Arrow_HWidth*np.cos(rA) ), rY, rY + ( Arrow_Length*np.sin(rA) + Arrow_HWidth*np.cos(rA) ) ]
        ]
        
        
    def update(this):
        super().update()
        this.graphRBody.set_data( this.robotPosRend[0], this.robotPosRend[1] )
        this.subAxis.draw_artist( this.graphRBody )
        
        this.graphPCloud.set_data( this.pointCloud[0], this.pointCloud[1] )
        this.subAxis.draw_artist( this.graphPCloud )
        
        

def debugExample():
    window = PlotWindow( 6, 4 )

    xTextDisp = ValueDisplay(0,0,1, window, "x")
    yTextDisp = ValueDisplay(0,1,1, window, "y")
    aTextDisp = ValueDisplay(0,2,1, window, "a")

    graphDisp = LineGraphDisplay( 1, 0, 2, 2, window, 1, -1, 1, -1 )
    
   # poseDisp = PoseDisplay( 0, 3, 2, 2, window, 5, -5, 5, -5 ) 
   
    roboDisp = RobotDisplay( 0, 3, 2, 3, window, 5, -5, 5, -5 )

    xVals = np.arange(-1, 1, 0.1)
    
    #window.render()

    for i in range(0, 2000):
        delay(0.05) 
        
        graphDisp.parseData( xVals, np.sin( xVals*3 + i/20 ) )
        
        nX, nY = 4*np.sin(i/10), 4*np.sin(i/12)
        pX, pY = 4*np.sin((i-1)/10), 4*np.sin((i-1)/12)
        nAlpha = np.arctan2(nY-pY, nX-pX)
        
        xTextDisp.parseData( nX )
        yTextDisp.parseData( nY )
        aTextDisp.parseData( nAlpha )
        
        roboDisp.parseData( CartesianPose( nX, nY, 0, 0, 0, nAlpha ), [[0,1,0,-2],[0,1,2,3]] )
               
        """poseDisp.parseData( [
            CartesianPose( nX, nY, 0, 0, 0, nAlpha )
        ] )"""
        
        window.render()
        
        print(i)
        
         







