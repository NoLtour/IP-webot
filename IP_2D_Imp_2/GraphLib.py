import g2o

import numpy as np

import plotly.graph_objects as go 

"""
    This is basically all taken from https://github.com/miquelmassot/g2o-python
"""

def plot_slam2d(optimizer, title):
    def edges_coord(edges, dim):
        for e in edges:
            yield e.vertices()[0].estimate()[dim]
            yield e.vertices()[1].estimate()[dim]
            yield None

    fig = go.Figure()

    # edges
    edges = optimizer.edges()  # get set once to have same order
    se2_edges = [e for e in edges if type(e) == g2o.EdgeSE2]
    se2_pointxy_edges = [e for e in edges if type(e) == g2o.EdgeSE2PointXY]
    
    
    fig.add_trace(
        go.Scatter(
            x=list(edges_coord(se2_pointxy_edges, 0)),
            y=list(edges_coord(se2_pointxy_edges, 1)),
            mode="lines",
            line=dict(
                color='firebrick', 
                width=2,
                dash='dash'),
            name="Measurement edges",
            legendgroup="Measurements"
            
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(edges_coord(se2_edges, 0)),
            y=list(edges_coord(se2_edges, 1)),
            mode="lines",
            line=dict(
                color='midnightblue', 
                width=4),
            name="Pose edges",
            legendgroup="Poses"
        )
    )
    
    # poses of the vertices
    vertices = optimizer.vertices()
    poses = [v.estimate() for v in vertices.values() if type(v) == g2o.VertexSE2]
    measurements = [v.estimate() for v in vertices.values() if type(v) == g2o.VertexPointXY]
    
    
    fig.add_trace(
        go.Scatter(
            x=[v[0] for v in poses],
            y=[v[1] for v in poses],
            mode="markers",
            marker_line_color="midnightblue", 
            marker_color="lightskyblue",
            marker_line_width=2, marker_size=15,
            name="Poses",
            legendgroup="Poses"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[v[0] for v in measurements],
            y=[v[1] for v in measurements],
            mode="markers",
            marker_symbol="star",
            marker_line_color="firebrick",
            marker_color="firebrick",
            marker_line_width=2, marker_size=15,
            name="Measurements",
            legendgroup="Measurements"
        )
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(go.Layout({"title": title}))

    return fig

class GraphSLAM2D:
    def __init__(this, verbose=False) -> None:
        '''
        GraphSLAM in 2D with G2O
        '''
        this.optimizer = g2o.SparseOptimizer()
        this.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        this.algorithm = g2o.OptimizationAlgorithmLevenberg(this.solver)
        this.optimizer.set_algorithm(this.algorithm)

        this.vertex_count = 0
        this.edge_count = 0
        this.verbose = verbose

    def plot(this, title=""):
        plot_slam2d( this.optimizer, title ).show()
    
    def vertex_pose(this, id):
        '''
        Get position of vertex by id
        '''
        return this.optimizer.vertex(id).estimate().to_vector()

    def vertex(this, id):
        '''
        Get vertex by id
        '''
        return this.optimizer.vertex(id)

    def edge(this, id):
        '''
        Get edge by id
        '''
        return this.optimizer.edge(id)

    def add_fixed_pose(this ):
        '''
        Add fixed pose to the graph
        '''
        pose = g2o.SE2()
        vertex_id=None
        v_se2 = g2o.VertexSE2()
        if vertex_id is None:
            vertex_id = this.vertex_count
        v_se2.set_id(vertex_id)
        if this.verbose:
            print("Adding fixed pose vertex with ID", vertex_id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(True)
        this.optimizer.add_vertex(v_se2)
        this.vertex_count += 1
        
        return vertex_id

    def add_pose( this, previousVertexID, offset:np.ndarray, strength=1 ):
        vertices = this.optimizer.vertices() 
        
        newPoseID = this.vertex_count
         
        
        g2PoseOffset = g2o.SE2(offset[0], offset[1], offset[2])
        
        v_se2 = g2o.VertexSE2()
        v_se2.set_id( newPoseID )
        #v_se2.set_estimate( g2PoseOffset )
        
        this.optimizer.add_vertex(v_se2)
        this.vertex_count += 1
        
        e_se2 = g2o.EdgeSE2()
        e_se2.set_vertex(0, this.vertex(previousVertexID))
        e_se2.set_vertex(1, this.vertex(newPoseID))
        e_se2.set_measurement( g2PoseOffset )
        e_se2.set_information( strength*np.eye(3) )
        
        this.optimizer.add_edge(e_se2)
        this.edge_count += 1
        
        return newPoseID

    def relate_pose( this, pose1ID, pose2ID, offset:np.ndarray, strength=1 ): 
         
        g2PoseOffset = g2o.SE2(offset[0], offset[1], offset[2])
        
        e_se2 = g2o.EdgeSE2()
        e_se2.set_vertex(0, this.vertex(pose1ID))
        e_se2.set_vertex(1, this.vertex(pose2ID))
        e_se2.set_measurement( g2PoseOffset )
        e_se2.set_information( strength*np.eye(3) )
        
        this.optimizer.add_edge(e_se2)
        this.edge_count += 1

    def optimize(this, iterations=10, verbose=None):
        '''
        Optimize the graph
        '''
        this.optimizer.initialize_optimization()
        if verbose is None:
            verbose = this.verbose
        #this.optimizer.set_verbose(verbose)
        this.optimizer.optimize(iterations)
        return this.optimizer.chi2()

    
""