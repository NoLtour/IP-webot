



from RawScanFrame import RawScanFrame

import matplotlib.pyplot as plt

print("importing...")
allScanData = RawScanFrame.importScanFrames( "cleanDataBackup" )
print("imported")

xB = []
yB = []

xR = []
yR = []

for cScan in allScanData:
    xB.append( cScan.pose.x )
    yB.append( cScan.pose.y )
    xR.append( cScan.truePose.x )
    yR.append( cScan.truePose.y )
 
plt.plot( xB, yB, "r-" ) 
plt.plot( xR, yR, "b-" )
plt.show()

""




