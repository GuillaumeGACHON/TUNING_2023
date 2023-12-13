import xarray as xr
import time
import numpy as np
from netCDF4 import Dataset
import glob
import seaborn
import sys
from dico_ds_to_sc import *
from dico_ds_to_map import *
from dash import Dash, html, dcc, Input, Output, Patch, State
import scipy.optimize as op
import scipy.stats as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from kapteyn import kmpfit

dset=xr.open_dataset("Metrics.nc")
dset_ts=xr.open_dataset("TimeSeries.nc")
params=xr.open_dataset("Params_TUN.nc")
dset_sc=xr.open_dataset("Seasonal.nc")
dset_map={"OCE_Grid_T_LR":xr.open_dataset("Climatos_OCE_Grid_T_LR.nc"),
          "OCE_Grid_T_DepthLv_LR":xr.open_dataset("Climatos_OCE_Grid_T_DepthLv_LR.nc"),
          "OCE_Diaptr_W_LR":xr.open_dataset("Climatos_OCE_Diaptr_W_LR.nc"),
          "ICE_LR":xr.open_dataset("Climatos_ICE_LR.nc"),
          "ATM_LR":xr.open_dataset("Climatos_ATM_LR.nc"),
          "OCE_Grid_T_VLR":xr.open_dataset("Climatos_OCE_Grid_T_VLR.nc"),
          "OCE_Grid_T_DepthLv_VLR":xr.open_dataset("Climatos_OCE_Grid_T_DepthLv_VLR.nc"),
          "OCE_Diaptr_W_VLR":xr.open_dataset("Climatos_OCE_Diaptr_W_VLR.nc"),
          "ICE_VLR":xr.open_dataset("Climatos_ICE_VLR.nc"),
          "ATM_VLR":xr.open_dataset("Climatos_ATM_VLR.nc")}




external_stylesheets = ["./assets/custom.css",'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__,external_stylesheets=external_stylesheets,prevent_initial_callbacks="initial_duplicate")

df = xr.combine_by_coords([dset,params])
df2 = dset_ts

def expon(xdata,A,B,C):
    return A*np.exp(B*xdata)+C

def regression(xdata,ydata,regtype):
    if regtype=="Linear":
        reg=st.linregress(xdata,ydata)
        coefs=[reg.intercept,reg.slope],reg.rvalue**2,reg.pvalue,[reg.intercept_stderr,reg.stderr]
    if regtype=="LinLog":
        reg=st.linregress(xdata,np.log(ydata-ydata.min))
        coefs=[reg.intercept,reg.slope,ydata.min],reg.rvalue**2,reg.pvalue,[reg.intercept_stderr,reg.stderr,np.nan]
    if regtype=="LogLin":
        reg=st.linregress(np.log(xdata-xdata.min),ydata)
        coefs=[reg.intercept,reg.slope],reg.rvalue**2,reg.pvalue,[reg.intercept_stderr,reg.stderr]
    if regtype=="LogLog":
        reg=st.linregress(np.log(xdata-xdata.min),np.log(ydata-ydata.min))
        coefs=[reg.intercept,reg.slope,ydata.min],reg.rvalue**2,reg.pvalue,[reg.intercept_stderr,reg.stderr,np.nan]
    if regtype=="Y=A*exp(B*X)+C":
        reg=op.curve_fit(expon,xdata,ydata,nan_policy="omit")
        coefs=reg.popt,np.nan,np.nan,np.sqrt(np.diag(reg.pcov))
    return coefs

#def reconstruction(xdata,coefs,regtype):
 #   if regtype=="Linear":
 #       y_fit=xdata*coefs[0][1]+coefs[0][0]
 #   if regtype=="LinLog":
 #       y_fit=coefs[0][2]+np.exp(coefs[0][1]*xdata+coefs[0][0])
 #   if regtype=="LogLin":
 #       y_fit=coefs[0][0]+coefs[0][1]*np.log(xdata-xdata.min)
 #   if regtype=="LogLog":
 #       y_fit=coefs[0][2]+np.exp(np.log(xdata-xdata.min)*coefs[0][1]+coefs[0][0])
 #   if regtype=="Y=A*exp(B*X)+C":
 #       y_fit=coefs[0][2]+coefs[0][0]*np.exp(coefs[0][1]*xdata)
 #   return y_fit

#def lower_bnd(xdata,coefs,regtype):
#    if regtype=="Linear":
#        y_fit=xdata*(coefs[0][1]-coefs[3][1])+coefs[0][0]-coefs[3][0]
#    if regtype=="LinLog":
#        y_fit=coefs[0][2]+np.exp((coefs[0][1]-coefs[3][1])*xdata+coefs[0][0]-coefs[3][0])
#    if regtype=="LogLin":
#        y_fit=coefs[0][0]-coefs[3][0]+(coefs[0][1]-coefs[3][1])*np.log(xdata-xdata.min)
#    if regtype=="LogLog":
#        y_fit=coefs[0][2]+np.exp(np.log(xdata-xdata.min)*(coefs[0][1]-coefs[3][1])+coefs[0][0]-coefs[3][0])
#    if regtype=="Y=A*exp(B*X)+C":
#        y_fit=coefs[0][2]+coefs[0][0]*np.exp(coefs[0][1]*xdata)
#    return y_fit
#
#def upper_bnd(xdata,coefs,regtype):
#    if regtype=="Linear":
#        y_fit=xdata*(coefs[0][1]-coefs[3][1])+coefs[0][0]-coefs[3][0]
#    if regtype=="LinLog":
#        y_fit=coefs[0][2]+np.exp((coefs[0][1]-coefs[3][1])*xdata+coefs[0][0]-coefs[3][0])
#    if regtype=="LogLin":
#        y_fit=coefs[0][0]-coefs[3][0]+(coefs[0][1]-coefs[3][1])*np.log(xdata-xdata.min)
#    if regtype=="LogLog":
#        y_fit=coefs[0][2]+np.exp(np.log(xdata-xdata.min)*(coefs[0][1]-coefs[3][1])+coefs[0][0]-coefs[3][0])
#    if regtype=="Y=A*exp(B*X)+C":
#        y_fit=coefs[0][2]+coefs[0][0]*np.exp(coefs[0][1]*xdata)
#    return y_fit


def model_expo(p, x):
    a, b, c = p
    return a*np.exp(b*x) +c

def model_lin(p,x):
    a,b = p
    return a*x +b

def model_poly2(p,x):
    a,b,c=p
    return a*x**2 +b*x +c

def model_poly3(p,x):
    a,b,c,d=p
    return a*x**3 +b*x**2 +c*x +d

def model_log(p,x):
    a,b,c=p
    return a*np.log(np.abs(x +b)) + c

def model_pow(p,x):
    a,b,c,d=p
    return a*np.abs(x +b)**c +d

def dfdp_fun(p,x,modelname):
    if modelname=="Linear":
        a,b=p
        return [x,1]
    if modelname=="Logarithmic":
        a,b,c=p
        return [np.log(np.abs(x+b)),a/np.abs(x+b),1]
    if modelname=="Exponential":
        a,b,c=p
        return [np.exp(b*x),a*x*np.exp(b*x),1]
    if modelname=="Power-like":
        a,b,c,d=p   
        return [np.abs(x+c)**b,a*np.abs(x+c)**b*np.log(np.abs(x+c)),np.sign(x+c)*a*b*np.abs(x+c)**(b-1),1]
    if modelname=="Polynomial 2":
        a,b,c=p
        return[x**2,x,1]
    if modelname=="Polynomial 3":
        a,b,c,d=p
        return [x**3,x**2,x,1]
        
models={"Exponential":model_expo,
        "Linear":model_lin,
        "Logarithmic":model_log,
        "Power-like":model_pow,
        "Polynomial 2":model_poly2,
        "Polynomial 3":model_poly3}

start_params={"Exponential":[.1,.1,.1],
        "Linear":[.1,.1],
        "Logarithmic":[.1,.1,.1],
        "Power-like":[.1,.1,.1,.1],
        "Polynomial 2":[.1,.1,.1],
        "Polynomial 3":[.1,.1,.1,.1]}

equations={"Exponential":"Y=A*exp(B*X)+C",
           "Linear":"Y=A*X+B",
           "Logarithmic":"Y=A*log|X + B|+C",
           "Power-like":"Y=A*|X+B|**C + D",
           "Polynomial 2":"Y=A*X**2+B*X +C",
           "Polynomial 3":"Y=A*X**3 + B*X**2 + C*X + D"}

def model_fit(modelname,x,y):
    f = kmpfit.simplefit(models[modelname], start_params[modelname], x, y)
    X=np.linspace(x.min(),x.max(),num=500)
    dfdp = dfdp_fun(f.params,X,modelname)
    return f.confidence_band(X, dfdp, 0.95, models[modelname]),f.params,f.stderr



coord_dict={"LR0":(0,"LR"),
	  "LR1":(1,"LR"),
	  "VLR0":(0,"VLR"),
	  "VLR1":(1,"VLR"),
	  "Hyb0":(0,"Hyb"),
	  "Hyb1":(1,"Hyb"),
	  "ICO":(0,"ICO")}
simcheck_index_Met={"LR0":0,"LR1":2,"VLR0":4,"VLR1":6,"ICO":8}
simcheck_index_TS={"LR0":1,"LR1":3,"VLR0":5,"VLR1":7,"ICO":9}


simcheck_index_Met2D_PPM={"LR0":0,"LR1":1,"VLR0":2,"VLR1":3,"ICO":4}
simcheck_index_TS_PPM={"LR0":1,"LR1":3,"VLR0":5,"VLR1":7,"ICO":9}
simcheck_index_Met3D_PPM={"LR0":0,"LR1":2,"VLR0":4,"VLR1":6,"ICO":8}

marker_dict1={"LR0":{"color":"royalblue","symbol":"circle"},
	  "LR1":{"color":"royalblue","symbol":"circle-open"},
	  "VLR0":{"color":"orangered","symbol":"square"},
	  "VLR1":{"color":"orangered","symbol":"square-open"},
	  "Hyb0":{"color":"seagreen","symbol":"diamond"},
	  "Hyb1":{"color":"seagreen","symbol":"diamond-open"},
	  "ICO":{"color":"darkorchid","symbol":"diamond"}}
marker_dict2={"LR0":{"size":4,"color":"black","symbol":"circle"},
	  "LR1":{"size":4,"color":"black","symbol":"circle-open"},
	  "VLR0":{"size":4,"color":"black","symbol":"square"},
	  "VLR1":{"size":4,"color":"black","symbol":"square-open"},
	  "Hyb0":{"size":4,"color":"black","symbol":"diamond"},
	  "Hyb1":{"size":4,"color":"black","symbol":"diamond-open"},
	  "ICO":{"size":4,"color":"black","symbol":"diamond"}}
marker_dict3={"LR0":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "LR1":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "VLR0":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "VLR1":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "Hyb0":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "Hyb1":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},
	  "ICO":{"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5}}
marker_dict4={"LR0":{"color":"royalblue","symbol":"circle"},
                  "LR1":{"line_color":"royalblue","symbol":"circle-open"},
                  "VLR0":{"color":"orangered","symbol":"square"},
                  "VLR1":{"line_color":"orangered","symbol":"square-open"},
                  "Hyb0":{"color":"seagreen","symbol":"diamond"},
                  "Hyb1":{"line_color":"seagreen","symbol":"diamond-open"},
                  "ICO":{"color":"darkorchid","symbol":"diamond"}}
line_colors={"LR0":{"color":"royalblue"},
                  "LR1":{"color":"royalblue",'dash':'dot'},
                  "VLR0":{"color":"orangered"},
                  "VLR1":{"color":"orangered",'dash':'dot'},
                  "Hyb0":{"color":"seagreen"},
                  "Hyb1":{"color":"seagreen",'dash':'dot'},
                  "ICO":{"color":"darkorchid"}}


figscatter=go.Figure()
figTSx=go.Figure()
figTSy=go.Figure()
figSC=go.Figure()
figMapX=go.Figure()
figMapY=go.Figure()
figMapZ=go.Figure()
figscatterPPM2D=go.Figure()
figscatterPPM3D=go.Figure()
figTSPPM=go.Figure()
figSCPPM=go.Figure()

for simcheck in ["LR0","LR1","VLR0","VLR1","ICO"]:
    etau,simutype=coord_dict[simcheck]
    figscatter.add_trace( go.Scatter(x=[],
        y=[],
        hovertext=[],
        opacity=1,
        marker=marker_dict1[simcheck],
        mode="markers",
        customdata=np.array([]), 
        name="Metrics "+simcheck
        ))
    figscatter.add_traces(go.Scatter(x=[],
        y=[],
        hoverinfo='skip',
        opacity=0.2,
        marker=marker_dict2[simcheck],
        mode="markers",
        showlegend=True,
        name="TS "+simcheck))
figscatter.add_trace(go.Scatter(x=[],y=[],name="Selected TS",hoverinfo='skip',opacity=1,marker={"size":4,"symbol":"cross-thin","line_color":"black","line_width":0.5},showlegend=False,line_width=0.6,line_color="black",line_dash="dot",mode="lines+markers"))
figscatter.add_trace(go.Scatter(x=[],y=[],name="Regression on Selected Data",hoverinfo='skip',opacity=1,line_width=1.5,line_color="black",mode="lines"))
figscatter.add_trace(go.Scatter(
        name='Upper Bound',
        x=[],
        y=[],
        hoverinfo='skip',
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False))
figscatter.add_trace(go.Scatter(
        name='Lower Bound',
        x=[],
        y=[],
        hoverinfo='skip',
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False))
figscatter.add_annotation(x=0, y=0.95, xanchor='left', yanchor='top',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=14)
figscatter.update_xaxes(title="", type='linear')
figscatter.update_yaxes(title="", type='linear')
figscatter.update_layout(font_family="Didot",height=800,margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest',legend_xref="paper",legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",legend_x=0.99)

from PIL import Image
img = Image.open('assets/Mevisto.png')

figscatter.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=-1,
            y=3.5,
            sizex=7,
            sizey=7,
#            sizing="stretch",
            opacity=0.7,
            layer="above")
)


figTSx.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=20)
figTSx.update_xaxes(showgrid=False)
figTSx.update_layout(legend_xref="paper",legend_x=0.99,legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right", font_family="Didot", xaxis_title="Years",yaxis_title="",margin={'l': 0, 'b': 30, 'r': 0, 't': 10})


figTSy.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=20)
figTSy.update_xaxes(showgrid=False)
figTSy.update_layout(legend_xref="paper",legend_x=0.99,legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",font_family="Didot", xaxis_title="Years",yaxis_title="",margin={'l': 0, 'b': 30, 'r': 0, 't': 10})


figMapX.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0},font_family='Didot')
figMapY.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0},font_family='Didot')
figMapX.add_annotation(x=0.3, y=0.5, xanchor='auto', yanchor='auto',
                       xref='paper', yref='paper', showarrow=False, align='center',
                       text="NO CLIMATOLOGY MAP",font_size=20)
figMapY.add_annotation(x=0.3, y=0.5, xanchor='auto', yanchor='auto',
                       xref='paper', yref='paper', showarrow=False, align='center',
                       text="NO CLIMATOLOGY MAP",font_size=20)
figMapZ.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0},font_family='Didot')
figMapZ.add_annotation(x=0.3, y=0.5, xanchor='auto', yanchor='auto',
                       xref='paper', yref='paper', showarrow=False, align='center',
                       text="NO CLIMATOLOGY MAP",font_size=20)

figSC.update_xaxes(title="", type='linear')
figSC.update_yaxes(title="", type='linear')
figSC.update_layout(font_family="Didot",height=800,margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, legend_xref="paper",legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",legend_x=0.99)
figSC.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=20)




figTSPPM.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=20)
figTSPPM.update_xaxes(showgrid=False)
figTSPPM.update_layout(legend_xref="paper",legend_x=0.99,legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right", font_family="Didot", xaxis_title="Years",yaxis_title="",margin={'l': 0, 'b': 30, 'r': 0, 't': 10})


figSCPPM.update_xaxes(title="", type='linear')

figSCPPM.update_layout(font_family="Didot",height=800,margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, legend_xref="paper",legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",legend_x=0.99)
figSCPPM.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="",font_size=20)

for simcheck in ["LR0","LR1","VLR0","VLR1","ICO"]:
    etau,simutype=coord_dict[simcheck]
    figscatterPPM3D.add_trace( go.Scatter3d(x=[],
        y=[],
        z=[],
        hovertext=[],
        opacity=1,
        marker=marker_dict1[simcheck],
        mode="markers",
        customdata=np.array([]),
        name="Metrics3D "+simcheck,
        marker_colorscale="Bluered"
        ))
    figscatterPPM3D.add_traces(go.Scatter3d(x=[],
        y=[],
        z=[],
        hoverinfo='skip',
        opacity=0.2,
        marker=marker_dict2[simcheck],
        mode="markers",
        showlegend=True,
        name="TS "+simcheck))
figscatterPPM3D.add_trace(go.Scatter3d(x=[],y=[],z=[],name="Selected TS",visible=False,hoverinfo='skip',opacity=1,marker={"size":4,"symbol":"x","line_color":"black","line_width":0.5},showlegend=False,line_width=0.6,line_color="black",line_dash="dot",mode="lines+markers"))



figscatterPPM3D.update_scenes(yaxis_title_text="",xaxis_title_text="",zaxis_title_text="")
figscatterPPM3D.update_xaxes(title="", type='linear')
figscatterPPM3D.update_yaxes(title="", type='linear')
figscatterPPM3D.update_layout(font_family="Didot",height=800,margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest',legend_xref="paper",legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",legend_x=0.99)

#print(figscatterPPM3D)

for simcheck in ["LR0","LR1","VLR0","VLR1","ICO"]:
    etau,simutype=coord_dict[simcheck]
    figscatterPPM2D.add_trace( go.Scatter(x=[],
        y=[],
        hovertext=[],
        opacity=1,
        marker=marker_dict1[simcheck],
        marker_size=8,
        mode="markers",
        customdata=np.array([]),
        marker_colorscale="Bluered",
        name="Metrics "+simcheck
        ))

figscatterPPM2D.update_xaxes(title="", type='linear')
figscatterPPM2D.update_yaxes(title="", type='linear')
figscatterPPM2D.update_layout(font_family="Didot",height=800, margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest',legend_xref="paper",legend_bgcolor='rgba(0,0,0,0)',legend_xanchor="right",legend_x=0.99)

#,scattermode="group", scattergap=0.00000001


app.layout = html.Div([html.Div([html.H6('Welcome to',style={'color': "black",'display':'inline-block','marginLeft':'120px'}), 
                                 html.H1('MeVisTo',style={'color': "darkred",'display':'inline-block','margin':'0px 30px'}), 
                                 html.H6('a MEtric VISualisation TOol',style={'color': "black",'display':'inline-block'})],
                 style={'textAlign': 'center','verticalAlign': 'top','fontFamily':"Didot",'color': "darkred"}),
    html.Div(style={'width':'100%'},children=[dcc.Tabs(id='maintab',value='tab-2D',
                                                       style={'width':'90vh','height':'3vw','transformOrigin':'left','transform':'translate(1vw,90vh) rotate(270deg)'},
                                                       children=[dcc.Tab(label='Param-Param-Metric Comparison',style={'padding':'2px 25px'},selected_style={'padding':'2px 25px','color':'darkred','borderTop':'darkred'},value='tab-3D',children=[html.Div([html.Div([
    html.Div([
            dcc.Dropdown(
                list(params),
                id='crossparam-xaxis-column',placeholder="Select for xaxis an AMIP_metric, a PARAMETER, or metric_REGION_Time"
                , maxHeight=400
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossparam-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                style={'display': 'inline-block'})
           ],
        style={'width': '47%', 'display': 'inline-block'}),
    html.Button("\u21CC",id="switch-axes-PPM",n_clicks=0,style={'height':'36px','fontSize':20,'textAlign': 'center','verticalAlign': 'top', 'width':'6%','padding':'0 0'}),
    html.Div([
            dcc.Dropdown(
                list(params),
                id='crossparam-yaxis-column',placeholder="Select a yaxis PARAMETER"
                , maxHeight=400
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossparam-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '47%', 'display': 'inline-block','verticalAlign':'top'})]
    , style={'padding': '10px 5px',"width":"49%",'display':'inline-block'}),
    html.Div([
    dcc.Dropdown(list(df),
                id='crossparam-metric-column',placeholder="Select a yaxis PARAMETER"
                , maxHeight=400,),
    dcc.Checklist(['Metric as Color','3D'],['Metric as Color'],id='crossparam-type',inline=True)
           ] ,style={'width':'49%','display':'inline-block','verticalAlign':'top','padding':'10px 5px'}),
    html.Div([html.Div([
        dcc.Graph(figure=figscatterPPM2D,
            id='crossparam-indicator-scatter2D',
            hoverData={'points': [{'customdata':np.array([])}]},
            style={'width': '100%', "verticalAlign": "top",'display': 'block', 'padding': '0 20','height':'49vw'}),
        dcc.Graph(figure=figscatterPPM3D,
            id='crossparam-indicator-scatter3D',
            hoverData={'points': [{'customdata':np.array([])}]},
            style={'width': '100%', "verticalAlign": "top",'display': 'none', 'padding': '0 20','height':'49vw'}),
    html.Div([dcc.Checklist(['LR0','Hyb0','VLR0'],[],id='crossparam-nnetau_0-check',inline=True),
              dcc.Checklist(['LR1','Hyb1','VLR1','ICO'],[],id='crossparam-nnetau_1-check',inline=True)],
              style={'width': '49%','padding': '0px 20px 20px 20px'})],
            style={'verticalAlign':'top','width': '49%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=figTSPPM,id='TS-PPM',style={"height": '30vh','display':'inline-block', 'width': '49%'}),
                  dcc.Graph(figure=figSCPPM,id='seasonal-PPM',style={"height":'30vh','width':'49%','display':'inline-block'}),
                  html.Div([dcc.RadioItems(['Raw', 'Anomaly','Standardised Anomaly'],'Raw',id='maps-PPM-flavor',style={'display': 'inline-block', 'width':'60%','marginTop': '5px'},labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
                              html.Button(children="Show Mean or STD Map",id="EnsMapsPPM",n_clicks=0,style={'backgroundColor':'#FFFFFF', 'width':'40%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}),
                              dcc.Graph(figure=figMapZ,id='maps-PPM',style={"width":"100%","height":'45vh',"verticalAlign": "top",'display': 'inline-block', 'padding': '0 0'}),
                              html.Div(id='text-zPPM',children="ColorScale",style={"width":"10%","display":"inline-block"}),
                              html.Button(children="AutoScale ON",id="ScalePPM",n_clicks=0,style={'backgroundColor':'#CCCCCC', 'width':'20%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}) ,
                              dcc.Input(id="input_minPPM",type="number",debounce=True,disabled=True,placeholder="Minimum Value",style={'width':'35%'}),dcc.Input(id="input_maxPPM",type="number",disabled=True,placeholder="Maximum Value",style={'width':'35%'}),
] ,style={'verticalAlign':'top', 'display': 'inline-block'})]
            ,style={'verticalAlign':'top','width': '49%', 'display': 'inline-block'})
    ], style={'width': '100%', "verticalAlign": "top",'display': 'inline-block', 'padding': '0 20'}
)
#, html.Div(dcc.Dropdown([],[],multi=True)
],style={'paddingRight':'3vw','transform':'translate(3vw,-2vw)','height':'90vh'})]),



dcc.Tab(label='ANY-ANY Comparison',selected_style={'padding':'2px 25px','color':'darkred','borderTop':'darkred'},style={'padding':'2px 25px'},value='tab-2D',children=[html.Div([html.Div([
    html.Div([
            dcc.Dropdown(
                list(df),
                id='crossfilter-xaxis-column',placeholder="Select for xaxis an AMIP_metric, a PARAMETER, or metric_REGION_Time"
                , maxHeight=400
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                style={'width': '30%', 'display': 'inline-block'}),
            html.Div(children=[
                html.Button("Regression",id="regonoff",n_clicks=0,style={'backgroundColor':'#FFFFFF', 'width':'40%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}),
                html.Div([dcc.Dropdown(["Linear","Exponential","Logarithmic","Power-like","Polynomial 2","Polynomial 3"],value="Linear",id='regtype',placeholder="Select your regression")],style={'width': '60%', 'display': 'inline-block', "verticalAlign": "top"})],
               style={'width':'70%','display': 'inline-block',"verticalAlign": "top"})
        ],
        style={'width': '47%', 'display': 'inline-block'}),
    html.Button("\u21CC",id="switch-axes",n_clicks=0,style={'height':'36px','fontSize':20,'textAlign': 'center','verticalAlign': 'top', 'width':'6%','padding':'0 0'}),
        html.Div([
            dcc.Dropdown(
                list(df),
                id='crossfilter-yaxis-column',placeholder="Select for yaxis an AMIP_metric, a PARAMETER, or Metric_REGION_Time"
                , maxHeight=400
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '47%', 'display': 'inline-block','verticalAlign':'top'})
    ], style={'padding': '10px 5px'}),
    html.Div([
        html.Div([dcc.Graph(figure=figscatter,
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata':np.array([])}]},style={"width":"100%","verticalAlign": "top",'display': 'inline-block', 'padding': '0 20','height':'49vw'}),
    html.Div([dcc.Checklist(['LR0','Hyb0','VLR0'],[],id='crossfilter-nnetau_0-check',inline=True),
              dcc.Checklist(['LR1','Hyb1','VLR1','ICO'],[],id='crossfilter-nnetau_1-check',inline=True)],
              style={'width': '60%','padding': '0px 20px 20px 20px'})],style={'width': '49%', "verticalAlign": "top",'display': 'inline-block'}),
#        ,style={'width': '49%', "verticalAlign": "top",'display': 'inline-block', 'padding': '0 20','height':'49vw'} ),     
        html.Div([dcc.Tabs(id="tabs-2D", 
                           value='tab-1', 
                           style={"height":30}, 
                           children=[
            dcc.Tab(label='Time Series', 
                    value='tab-1',
                    style={"padding":"3px 25px"},
                    selected_style={"padding":"3px 20px",'color':'darkred','borderTopColor':'darkred'},
                    children=[html.Div([
                                     dcc.Graph(figure=figTSx,id='x-time-series',style={"height": '24vw'}),
                                     dcc.Graph(figure=figTSy,id='y-time-series',style={"height": '24vw'}),
                                       ], style={'display':'inline-block', 'width': '100%'})]
                    ),
            dcc.Tab(label='Mean Seasonal Cycle', 
                    value='tab-2',
                    style={"padding":"3px 25px"},
                    selected_style={"padding":"3px 20px",'color':'darkred','borderTopColor':'darkred'},
                    children=[dcc.Graph(figure=figSC,id='seasonal',style={"height":'48vw'})]
                    ),
            dcc.Tab(label='Climatological Maps', 
                    value='tab-3',
                    style={"padding":"3px 25px"},
                    selected_style={"padding":"3px 20px",'color':'darkred','borderTopColor':'darkred'},
                    children=[dcc.RadioItems(['Raw', 'Anomaly','Standardised Anomaly'],'Raw',id='maps-flavor',style={'display': 'inline-block', 'width':'60%','marginTop': '5px'},labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
                              html.Button(children="Show Mean or STD Map",id="EnsMaps",n_clicks=0,style={'backgroundColor':'#FFFFFF', 'width':'40%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}),
                              dcc.Graph(figure=figMapX,id='maps-x',style={"width":"100%","height":'21.9vw',"verticalAlign": "top",'display': 'inline-block', 'padding': '0 0'}),
                              dcc.Graph(figure=figMapY,id='maps-y',style={"width":"100%","height":'21.9vw',"verticalAlign": "top",'display': 'inline-block', 'padding': '0 0'}),
                              html.Div(id='text-x',children="X ColorScale",style={"width":"15%","display":"inline-block"}),
                              html.Button(children="AutoScale ON",id="ScaleX",n_clicks=0,style={'backgroundColor':'#CCCCCC', 'width':'20%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}) ,
                              dcc.Input(id="input_minx",type="number",debounce=True,disabled=True,placeholder="Minimum Value",style={'width':'32.5%'}),dcc.Input(id="input_maxx",type="number",disabled=True,placeholder="Maximum Value",style={'width':'32.5%'}),
                              html.Div(id='text-y',children="Y ColorScale",style={'width':'15%',"display":"inline-block"}),
                              html.Button(children="AutoScale ON",id="ScaleY",n_clicks=0,style={'backgroundColor':'#CCCCCC', 'width':'20%', 'height':'36px', 'padding':'0 1vw','textAlign':'center', 'display': 'inline-block', "verticalAlign": "top"}) ,
                              dcc.Input(id="input_miny",type="number",debounce=True,disabled=True,placeholder="Minimum Value",style={'width':'32.5%'}),dcc.Input(id="input_maxy",type="number",disabled=True,placeholder="Maximum Value",style={'width':'32.5%'}),
]
                   )
            ], )],style={'width': '49%', 'display': 'inline-block'})
    ], style={'width': '100%', "verticalAlign": "top",'display': 'inline-block', 'padding': '0 20'}
),
 
],style={'paddingRight':'3vw','transform':'translate(3vw,-2vw)','height':'90vh'})])])])])



 
### 3D Scatter Plot



@app.callback(
    Output('crossparam-xaxis-column', 'value'),
    Output('crossparam-yaxis-column', 'value'),
    State('crossparam-xaxis-column', 'value'),
    State('crossparam-yaxis-column', 'value'),
    Input('switch-axes-PPM','n_clicks'))
def switch_axes_PPM(xaxis,yaxis,click):
    return yaxis,xaxis



@app.callback(
    Output('crossparam-indicator-scatter2D', 'figure',allow_duplicate=True),
    Output('crossparam-indicator-scatter3D', 'figure', allow_duplicate=True),
    Output('crossparam-indicator-scatter2D', 'style', allow_duplicate=True),
    Output('crossparam-indicator-scatter3D', 'style',allow_duplicate=True),
    Input('crossparam-xaxis-column', 'value'),
    Input('crossparam-yaxis-column', 'value'),
    Input('crossparam-metric-column', 'value'),
    Input('crossparam-nnetau_0-check', 'value'),
    Input('crossparam-nnetau_1-check','value'),
    Input('crossparam-type','value'))
def update_BG_PPM(xaxis_column_name, yaxis_column_name,metric_column_name,
                 nnetau0,nnetau1,crossparam_type,selpoints=[]):
    styleon=style={'width': '100%', "verticalAlign": "top",'display': 'block', 'padding': '0 20','height':'49vw'}
    styleoff=style={'width': '100%', "verticalAlign": "top",'display': 'none', 'padding': '0 20','height':'49vw'}
    
    style2D=styleon
    style3D=styleoff
    dff = df
    fig2D = Patch()
    fig3D=Patch()
    if xaxis_column_name!=None and yaxis_column_name!=None and metric_column_name!=None and nnetau0+nnetau1!=[]:
        fig3D.layout.images=[]
    for simcheck in ["LR0","LR1","VLR0","VLR1","ICO"]:
        if simcheck in (nnetau0+nnetau1):
            etau,simutype=coord_dict[simcheck]
            dset=dff.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
            print(dset["NSimu"].data)
            dset2=df2.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
            custom=np.stack([np.array([simutype]*len(dset['NSimu'])),np.array([etau]*len(dset['NSimu'])),dset['NSimu'].data],axis=-1)
            print("Custom :",custom)
            print("Selected Vars : X =",xaxis_column_name,", Y=",yaxis_column_name)

            if  xaxis_column_name==None or yaxis_column_name==None or metric_column_name==None:    
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["x"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["y"]=[] 
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["z"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["hovertext"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["customdata"]=np.array([])

                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["x"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["y"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["hovertext"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["customdata"]=np.array([])
            elif "3D" in crossparam_type:
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["x"]=dset[xaxis_column_name].data
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["y"]=dset[yaxis_column_name].data
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["z"]=dset[metric_column_name].data
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["hovertext"]=dset['Simu_Name'].data
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["customdata"]=custom

                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["x"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["y"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["hovertext"]=[]
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["customdata"]=np.array([])
         
                style2D=styleoff
                style3D=styleon
            else:
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["x"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["y"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["z"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["hovertext"]=[]
                fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["customdata"]=np.array([])

                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["x"]=dset[xaxis_column_name].data
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["y"]=dset[yaxis_column_name].data
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["hovertext"]=dset['Simu_Name'].data
                fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["customdata"]=custom       
           

                style2D=styleon
                style3D=styleoff
            if "Metric as Color" in crossparam_type:
                print("Coloring")
                if "3D" in crossparam_type:
                    fig3D.data[simcheck_index_Met3D_PPM[simcheck]].marker.color=dset[metric_column_name].data
                else:
                    fig2D.data[simcheck_index_Met2D_PPM[simcheck]].marker.color=dset[metric_column_name].data
            else:
                fig3D.data[simcheck_index_Met3D_PPM[simcheck]].marker.color=marker_dict1[simcheck]["color"]
                fig2D.data[simcheck_index_Met2D_PPM[simcheck]].maker.color=marker_dict1[simcheck]["color"]
            if xaxis_column_name==None or yaxis_column_name==None or metric_column_name==None or not "3D" in crossparam_type:
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["x"]=[]
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["y"]=[]
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["z"]=[]
            else:
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["x"]=np.tile(dset[xaxis_column_name].data.T,(dset2.year.size,1)).T.ravel()
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["z"]=dset2[metric_column_name].data.ravel() 
                fig3D["data"][simcheck_index_TS_PPM[simcheck]]["y"]=np.tile(dset[yaxis_column_name].data.T,(dset2.year.size,1)).T.ravel()
        else:
            fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["x"]=[]
            fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["y"]=[]
            fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["z"]=[]
            fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["hovertext"]=[]
            fig3D["data"][simcheck_index_Met3D_PPM[simcheck]]["customdata"]=np.array([])
            fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["x"]=[]
            fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["y"]=[]
            fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["hovertext"]=[]
            fig2D["data"][simcheck_index_Met2D_PPM[simcheck]]["customdata"]=np.array([])
            fig3D.data[simcheck_index_Met3D_PPM[simcheck]].marker.color=marker_dict1[simcheck]["color"]
            fig2D.data[simcheck_index_Met2D_PPM[simcheck]].marker.color=marker_dict1[simcheck]["color"]
            fig3D["data"][simcheck_index_TS_PPM[simcheck]]["x"]=[]
            fig3D["data"][simcheck_index_TS_PPM[simcheck]]["y"]=[]
    return fig2D,fig3D,style2D,style3D




@app.callback(
    Output('crossparam-indicator-scatter3D', 'figure',allow_duplicate=True),
    Input('crossparam-xaxis-column', 'value'),
    Input('crossparam-yaxis-column', 'value'),
    Input('crossparam-metric-column', 'value'),
    Input('crossparam-type','value'),
    Input('crossparam-indicator-scatter3D', 'hoverData'))
def update_selected(xaxis_column_name, yaxis_column_name,metric_column_name,crossparam_type,hoverData,selpoints=[]):
    dff = df
    simu_coord = hoverData['points'][0]['customdata']
    fig = Patch()
#    if simu_coord==np.array([]):
#        return fig
    fig.data[10].line=line_colors[simu_coord[0]+simu_coord[1]]
    if xaxis_column_name==None or yaxis_column_name==None or metric_column_name==None or not "3D" in crossparam_type:
        fig["data"][10]["x"]=[]
        fig["data"][10]["y"]=[]
        fig["data"][10]["z"]=[]
    else:
        fig["data"][10]["x"]=np.tile(df[xaxis_column_name].sel(NSimu=int(simu_coord[2])).data,df2.year.size)
        fig["data"][10]["y"]=np.tile(df[yaxis_column_name].sel(NSimu=int(simu_coord[2])).data,df2.year.size)
        fig["data"][10]["z"]=df2[metric_column_name].sel(NSimu=int(simu_coord[2]),SimuType=simu_coord[0],nnetau=int(simu_coord[1])).data
    return fig


@app.callback(
    Output('crossparam-indicator-scatter2D', 'figure',allow_duplicate=True),
    Output('crossparam-indicator-scatter3D', 'figure',allow_duplicate=True),
    Input('crossparam-xaxis-column', 'value'),
    Input('crossparam-yaxis-column', 'value'),
    Input('crossparam-metric-column', 'value'),
    Input('crossparam-xaxis-type', 'value'),
    Input('crossparam-yaxis-type', 'value'))
def update_layout(xaxis_column_name, yaxis_column_name,metric_column_name,
                 xaxis_type, yaxis_type):
    fig2D = Patch()
    fig3D = Patch()
    fig2D["layout"]["xaxis"]["title"]=xaxis_column_name
    fig2D["layout"]["xaxis"]["type"]='linear' if xaxis_type == 'Linear' else 'log'
    fig2D["layout"]["yaxis"]["title"]=yaxis_column_name
    fig2D["layout"]["yaxis"]["type"]='linear' if yaxis_type == 'Linear' else 'log'
    fig3D["layout"]["xaxis"]["title"]=xaxis_column_name
    fig3D["layout"]["xaxis"]["type"]='linear' if xaxis_type == 'Linear' else 'log'
    fig3D["layout"]["yaxis"]["title"]=yaxis_column_name
    fig3D["layout"]["yaxis"]["type"]='linear' if yaxis_type == 'Linear' else 'log'
    fig3D.layout.scene.xaxis.title.text=xaxis_column_name
    fig3D.layout.scene.yaxis.title.text=yaxis_column_name
    fig3D.layout.scene.zaxis.title.text=metric_column_name
    return fig2D,fig3D



### 3D Time Series



@app.callback(
    Output('TS-PPM', 'figure',allow_duplicate=True),
    Input('crossparam-metric-column', 'value'),
#    State('crossparam-metric-type', 'value'),
    Input('crossparam-nnetau_0-check', 'value'),
    Input('crossparam-nnetau_1-check','value'))
def update_all_metric_timeseries(metric_column_name,nnetau0,nnetau1 ,axis_type='Linear'):
    if metric_column_name!=None and metric_column_name in df2:
        title = '<b>{}</b>'.format("No Sim Selected")
        return create_time_series(axis_type, title, metric_column_name,nnetau0+nnetau1)
    return empty_time_series()

@app.callback(
    Output('TS-PPM', 'figure',allow_duplicate=True),
    Input('crossparam-indicator-scatter2D', 'hoverData'),
    Input('crossparam-indicator-scatter3D', 'hoverData'),
    Input('crossparam-type', 'value'),
    Input('crossparam-metric-column', 'value'))
def update_selected_metric_timeseries(hoverData2D,hoverData3D,crossparam_type, metric_column_name):
    if "3D" in crossparam_type:
        hoverData=hoverData3D
    else: 
        hoverData=hoverData2D
    simu_custom=hoverData['points'][0]['customdata']
#    if simu_custom==np.array([]):
#        return Patch()
    simu_chk=simu_custom[0]+simu_custom[1]
    dff = df2.sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1]))
    if metric_column_name!=None and metric_column_name in df2:
        dff2 = dff[metric_column_name]
        title = '<b>{}</b>'.format(dff.Simu_Name.data)
        return selected_time_series(dff2,title,simu_chk)
    return Patch()




### 3D   Seasonal Cycles




def create_seasonal_PPM(title, metric_column_name,simucateg):
    fig  = Patch()
    data=[]
    for simcheck in simucateg:
        etau,simutype=coord_dict[simcheck]
        dset2=dset_sc.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
        for ind in range(len(dset2[metric_column_name])):
             data.append(go.Scatterpolar(mode='lines+markers',name="",connectgaps=True,r=recollement(dset2[metric_column_name][ind].data-np.nanmean(dset2[metric_column_name][ind].data)),theta=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"],showlegend=False,opacity=0.1,hoverinfo='skip',marker={"size":3,"symbol":0,"color":"black"},line_color="black"))

    data.append(go.Scatterpolar(mode='lines+markers',name="Sim",r=[], theta=[],text=[],textposition=f'top right',textfont={'size':30,'color':['white','black','white','black','white','black','white','black','white','black','white','black','white']},marker=marker_dict4["LR0"],line=line_colors["LR0"]))
#    fig.layout.yaxis.type='linear' if yaxis_type == 'Linear' else 'log'
#    fig.layout.xaxis.type='linear' if xaxis_type == 'Linear' else 'log'
    fig["layout"]["annotations"][0].x=0
    fig["layout"]["annotations"][0].y=0.85
    fig["layout"]["annotations"][0].xanchor='left'
    fig["layout"]["annotations"][0].yanchor='bottom'
    fig["layout"]["annotations"][0].align='left'
    fig["layout"]["annotations"][0].text=title
    fig["layout"]["annotations"][0].font_size=16
    fig.data=data
    return fig

def selected_seasonal_PPM(dff1,title,simu_chk):
    fig  = Patch() 
    fig["data"][-1]['r']=recollement(dff1.data-np.nanmean(dff1.data))
    fig["data"][-1]['theta']=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"]
    fig["data"][-1]["marker"]=marker_dict4[simu_chk]
    fig["data"][-1]["line"]=line_colors[simu_chk]
    fig["data"][-1]["name"]=title
    fig["layout"]["annotations"][0]["text"]=title 
    return fig


@app.callback(
    Output('seasonal-PPM', 'figure',allow_duplicate=True),
    Input('crossparam-metric-column', 'value'),
    Input('crossparam-nnetau_0-check', 'value'),
    Input('crossparam-nnetau_1-check','value'))
def update_all_seasonal_PPM(metric_column_name, nnetau0,nnetau1):
    if metric_column_name!=None and Met_to_SC[metric_column_name] in dset_sc :
        title = '<b>{}</b>'.format("No Sim Selected")
        return create_seasonal_PPM(title, Met_to_SC[metric_column_name],nnetau0+nnetau1)
    return empty_seasonal()

@app.callback(
    Output('seasonal-PPM', 'figure',allow_duplicate=True),
    Input('crossparam-indicator-scatter2D', 'hoverData'),
    Input('crossparam-indicator-scatter3D', 'hoverData'),
    Input('crossparam-type', 'value'),
    Input('crossparam-metric-column', 'value'))
def update_selected_seasonal_PPM(hoverData2D,hoverData3D,crossparam_type, metric_column_name):
    if "3D" in crossparam_type:
        hoverData=hoverData3D
    else:
        hoverData=hoverData2D
    simu_custom=hoverData['points'][0]['customdata']
#    if simu_custom==np.array([]):
#        return Patch()
    simu_chk=simu_custom[0]+simu_custom[1]
    dff = dset_sc.sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1]))
    if metric_column_name!=None and Met_to_SC[metric_column_name] in dset_sc:
        dff1 = dff[Met_to_SC[metric_column_name]]
        title = '<b>{}</b>'.format(dff.Simu_Name.data)
        return selected_seasonal_PPM(dff1,title,simu_chk)
    return Patch()





### 2D Scatter Plot




@app.callback(
    Output('crossfilter-xaxis-column', 'value'),
    Output('crossfilter-yaxis-column', 'value'),
    State('crossfilter-xaxis-column', 'value'),
    State('crossfilter-yaxis-column', 'value'),
    Input('switch-axes','n_clicks'))
def switch_axes(xaxis,yaxis,click):
    return yaxis,xaxis



@app.callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-nnetau_0-check', 'value'),
    Input('crossfilter-nnetau_1-check','value'))
def update_BG(xaxis_column_name, yaxis_column_name,
                 nnetau0,nnetau1,selpoints=[]):
    dff = df
    fig = Patch()
    if xaxis_column_name!=None and yaxis_column_name!=None and nnetau0+nnetau1!=[]:
        fig.layout.images=[]
    for simcheck in ["LR0","LR1","VLR0","VLR1","ICO"]:
        if simcheck in (nnetau0+nnetau1):
            etau,simutype=coord_dict[simcheck]
            dset=dff.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
            print(dset["NSimu"].data)
            dset2=df2.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
            custom=np.stack([np.array([simutype]*len(dset['NSimu'])),np.array([etau]*len(dset['NSimu'])),dset['NSimu'].data],axis=-1)
            print("Custom :",custom)
            print("Selected Vars : X =",xaxis_column_name,", Y=",yaxis_column_name)

            if  xaxis_column_name==None or yaxis_column_name==None:    
                fig["data"][simcheck_index_Met[simcheck]]["x"]=[]
                fig["data"][simcheck_index_Met[simcheck]]["y"]=[]
                fig["data"][simcheck_index_Met[simcheck]]["hovertext"]=[]
                fig["data"][simcheck_index_Met[simcheck]]["customdata"]=np.array([])
            else:
                fig["data"][simcheck_index_Met[simcheck]]["x"]=dset[xaxis_column_name].data
                fig["data"][simcheck_index_Met[simcheck]]["y"]=dset[yaxis_column_name].data
                fig["data"][simcheck_index_Met[simcheck]]["hovertext"]=dset['Simu_Name'].data
                fig["data"][simcheck_index_Met[simcheck]]["customdata"]=custom
			
            if xaxis_column_name==None or yaxis_column_name==None or ((not xaxis_column_name in dset2) and (not yaxis_column_name in dset2)):
                fig["data"][simcheck_index_TS[simcheck]]["x"]=[]
                fig["data"][simcheck_index_TS[simcheck]]["y"]=[]
            elif not xaxis_column_name in dset2:
                fig["data"][simcheck_index_TS[simcheck]]["x"]=np.tile(dset[xaxis_column_name].data.T,(dset2.year.size,1)).T.ravel()
                fig["data"][simcheck_index_TS[simcheck]]["y"]=dset2[yaxis_column_name].data.ravel() 
            elif not yaxis_column_name in dset2:
                fig["data"][simcheck_index_TS[simcheck]]["x"]=dset2[xaxis_column_name].data.ravel()
                fig["data"][simcheck_index_TS[simcheck]]["y"]=np.tile(dset[yaxis_column_name].data.T,(dset2.year.size,1)).T.ravel()
            else:
                fig["data"][simcheck_index_TS[simcheck]]["x"]=dset2[xaxis_column_name].data.ravel()
                fig["data"][simcheck_index_TS[simcheck]]["y"]=dset2[yaxis_column_name].data.ravel()
        else:
            fig["data"][simcheck_index_Met[simcheck]]["x"]=[]
            fig["data"][simcheck_index_Met[simcheck]]["y"]=[]
            fig["data"][simcheck_index_Met[simcheck]]["hovertext"]=[]
            fig["data"][simcheck_index_Met[simcheck]]["customdata"]=np.array([])
            fig["data"][simcheck_index_TS[simcheck]]["x"]=[]
            fig["data"][simcheck_index_TS[simcheck]]["y"]=[]
    return fig




@app.callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'))
def update_selected(xaxis_column_name, yaxis_column_name,hoverData,selpoints=[]):
    dff = df
    simu_coord = hoverData['points'][0]['customdata']
    fig = Patch()
#    if simu_coord==np.array([]):
#        return fig
    fig.data[10].line=line_colors[simu_coord[0]+simu_coord[1]]
    if xaxis_column_name==None or yaxis_column_name==None or ((not xaxis_column_name in df2) and (not yaxis_column_name in df2)):
        fig["data"][10]["x"]=[]
        fig["data"][10]["y"]=[]
    elif not xaxis_column_name in df2:
        fig["data"][10]["x"]=np.tile(df[xaxis_column_name].sel(NSimu=int(simu_coord[2])).data,df2.year.size)
        fig["data"][10]["y"]=df2[yaxis_column_name].sel(NSimu=int(simu_coord[2]),SimuType=simu_coord[0],nnetau=int(simu_coord[1])).data
    elif not yaxis_column_name in df2:
        fig["data"][10]["x"]=df2[xaxis_column_name].sel(NSimu=int(simu_coord[2]),SimuType=simu_coord[0],nnetau=int(simu_coord[1])).data
        fig["data"][10]["y"]=np.tile(df[yaxis_column_name].sel(NSimu=int(simu_coord[2])).data,df2.year.size)
    else:
        fig["data"][10]["x"]=df2[xaxis_column_name].sel(NSimu=int(simu_coord[2]),SimuType=simu_coord[0],nnetau=int(simu_coord[1])).data
        fig["data"][10]["y"]=df2[yaxis_column_name].sel(NSimu=int(simu_coord[2]),SimuType=simu_coord[0],nnetau=int(simu_coord[1])).data
    return fig


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Input('regonoff','n_clicks'),
    Input('regtype','value'),
    State('crossfilter-xaxis-column', 'value'),
    State('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-nnetau_0-check', 'value'),
    Input('crossfilter-nnetau_1-check','value'))
def update_regression(click,regtype,xaxis_column_name, yaxis_column_name,nnetau0,nnetau1,selpoints=[]):
    if click%2==0:
        fig = Patch()
        fig["data"][11]["x"]=[]
        fig["data"][11]["y"]=[]
        fig["data"][11]["x"]=[]
        fig["data"][11]["y"]=[]
        fig["data"][12]["x"]=[]
        fig["data"][12]["y"]=[]
        fig.data[13].x=[]
        fig.data[13].y=[]
        fig["layout"]["annotations"][0].text=""
        return fig
    simtypes=nnetau0+nnetau1
    fig = Patch()
    fig.data[11].line={"color":"darkorchid"}
    if xaxis_column_name==None or yaxis_column_name==None:
        fig["data"][11]["x"]=[]
        fig["data"][11]["y"]=[]
    else:
        dff1=np.array([])
        dff2=np.array([])
        params=xr.open_dataset("Params_TUN.nc")
        if xaxis_column_name in params and yaxis_column_name in params:
            dff1 = df[xaxis_column_name].data
            dff2 = df[yaxis_column_name].data
        elif xaxis_column_name in params:
            for simtype in simtypes:
                etau,simutype=coord_dict[simtype]
                dff1 = np.concatenate((dff1,df[xaxis_column_name].data))
                dff2 = np.concatenate((dff2,df[yaxis_column_name].sel(SimuType=simutype,nnetau=etau).data))
        elif yaxis_column_name in params:
            for simtype in simtypes:
                etau,simutype=coord_dict[simtype]
                dff1 = np.concatenate((dff1,df[xaxis_column_name].sel(SimuType=simutype,nnetau=etau).data))
                dff2 = np.concatenate((dff2,df[yaxis_column_name].data))
        else:
            for simtype in simtypes:
                etau,simutype=coord_dict[simtype]
                dff1 = np.concatenate((dff1,df[xaxis_column_name].sel(SimuType=simutype,nnetau=etau).data))
                dff2 = np.concatenate((dff2,df[yaxis_column_name].sel(SimuType=simutype,nnetau=etau).data))
        print(dff1,"\n\n",dff2)
        print(dff1.min(),dff1.max())
        X=np.linspace(np.nanmin(dff1),np.nanmax(dff1),num=500)
        fig["data"][11]["x"]=X
#        reg=regression(dff1,dff2,regtype)
#        rec=reconstruction(X,reg,regtype)
        print(dff1[np.logical_not(np.isnan(dff1+dff2))])
        (rec,upper,lower),params,stderr=model_fit(regtype,dff1[np.logical_not(np.isnan(dff1+dff2))],dff2[np.logical_not(np.isnan(dff1+dff2))])
        print(params)
        fig["data"][11]["y"]=rec
        fig["data"][12]["x"]=X
        fig["data"][12]["y"]=upper
        fig.data[13].x=X
        fig.data[13].y=lower
        titre="   "+equations[regtype]+"<br>"
        param_let="ABCDEFG"
        for num in range(len(params)):
            titre+="-  %s = %.3f ± %.3f <br>"%(param_let[num],params[num],stderr[num])
        fig["layout"]["annotations"][0].text=titre
    return fig

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Output('regonoff','n_clicks'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'))
def clear_regression(xaxis_column_name,yaxis_column_name):
    fig = Patch()
    fig["data"][11]["x"]=[]
    fig["data"][11]["y"]=[]
    fig["data"][11]["x"]=[]
    fig["data"][11]["y"]=[]
    fig["data"][12]["x"]=[]
    fig["data"][12]["y"]=[]
    fig.data[13].x=[]
    fig.data[13].y=[]
    fig["layout"]["annotations"][0].text=""
    return fig,0


@app.callback(
    Output('regonoff','style'),
    Input('regonoff','n_clicks'))
def button_on_off(click):
    if click%2==0:
        return {'backgroundColor':'#FFFFFF', 'padding':'0 1vw', 'width': '40%', 'display': 'inline-block', "verticalAlign": "top" ,'height':'36px', 'textAlign':'center'}
    else:
        return {'backgroundColor':'#CCCCCC', 'padding':'0 1vw', 'width': '40%', 'display': 'inline-block',  'height':'36px', 'textAlign':'center', "verticalAlign": "top"}


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_layout(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):
    fig = Patch()
    fig["layout"]["xaxis"]["title"]=xaxis_column_name
    fig["layout"]["xaxis"]["type"]='linear' if xaxis_type == 'Linear' else 'log'
    fig["layout"]["yaxis"]["title"]=yaxis_column_name
    fig["layout"]["yaxis"]["type"]='linear' if yaxis_type == 'Linear' else 'log'
    return fig



### 2D Time Series




def create_time_series(axis_type, title, column_name,simucateg):
    fig  = Patch()
    data=[]
    for simcheck in simucateg:
        etau,simutype=coord_dict[simcheck]
        dset2=df2.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
        for ind in range(len(dset2[column_name])):
             data.append(go.Scatter(mode='lines+markers',name="",x=df2.year.data,y=dset2[column_name][ind].data,showlegend=False,opacity=0.1,hoverinfo='skip',marker={"size":3,"symbol":0,"color":"black"},line_color="black"))

    data.append(go.Scatter(mode='lines+markers',name="Sim",x=[], y=[],marker=marker_dict4["LR0"],line=line_colors["LR0"]))
    fig.layout.yaxis.type='linear' if axis_type == 'Linear' else 'log'
    fig["layout"]["annotations"][0].x=0
    fig["layout"]["annotations"][0].y=0.85
    fig["layout"]["annotations"][0].xanchor='left'
    fig["layout"]["annotations"][0].yanchor='bottom'
    fig["layout"]["annotations"][0].align='left'
    fig["layout"]["annotations"][0].text=title
    fig["layout"]["annotations"][0].font_size=16
    fig.layout.yaxis.title=column_name
    fig.data=data
    return fig

def selected_time_series(dff,title,simu_chk):
    fig  = Patch() 
    fig["data"][-1]['x']=dff.year.data
    fig["data"][-1]['y']=dff.data
    fig["data"][-1]["marker"]=marker_dict4[simu_chk]
    fig["data"][-1]["line"]=line_colors[simu_chk]
    fig["data"][-1]["name"]=title
    fig["layout"]["annotations"][0]["text"]=title 
    return fig

def empty_time_series():
    fig=Patch()
    fig["data"]=[]
    fig.layout.yaxis.title=""
    fig.layout["annotations"][0].x=0.3
    fig.layout["annotations"][0].y=0.5
    fig.layout["annotations"][0].xanchor='auto'
    fig.layout["annotations"][0].yanchor='auto'
    fig.layout["annotations"][0].align='center'
    fig.layout["annotations"][0].text="NO  TIME  SERIES"
    fig.layout["annotations"][0].font_size=55
    return fig

@app.callback(
    Output('x-time-series', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-xaxis-column', 'value'),
    State('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-nnetau_0-check', 'value'),
    Input('crossfilter-nnetau_1-check','value'))
def update_all_x_timeseries(tab_selected,xaxis_column_name, axis_type,nnetau0,nnetau1):
    if tab_selected!='tab-1':
        return Patch()
    if xaxis_column_name!=None and xaxis_column_name in df2:
        title = '<b>{}</b>'.format("No Sim Selected")
        return create_time_series(axis_type, title, xaxis_column_name,nnetau0+nnetau1)
    return empty_time_series()

@app.callback(
    Output('x-time-series', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'))
def update_selected_x_timeseries(tab_selected,hoverData, xaxis_column_name):
    if tab_selected!='tab-1':
        return Patch()
    simu_custom=hoverData['points'][0]['customdata']
#    if simu_custom==np.array([]):
#        return Patch()
    simu_chk=simu_custom[0]+simu_custom[1]
    dff = df2.sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1]))
    if xaxis_column_name!=None and xaxis_column_name in df2:
        dff2 = dff[xaxis_column_name]
        title = '<b>{}</b>'.format(dff.Simu_Name.data)
        return selected_time_series(dff2,title,simu_chk)
    return Patch()

@app.callback(
    Output('y-time-series', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-nnetau_0-check', 'value'),
    Input('crossfilter-nnetau_1-check','value'))
def update_all_y_timeseries(tab_selected,yaxis_column_name, axis_type,nnetau0,nnetau1):
    if tab_selected!='tab-1':
        return Patch()
    if yaxis_column_name!=None and yaxis_column_name in df2:
        title = '<b>{}</b>'.format("No Sim Selected")
        return create_time_series(axis_type, title, yaxis_column_name,nnetau0+nnetau1)
    return empty_time_series()


@app.callback(
    Output('y-time-series', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'))
def update_selected_y_timeseries(tab_selected,hoverData, yaxis_column_name):
    if tab_selected!='tab-1':
        return Patch()
    simu_custom=hoverData['points'][0]['customdata']
#    if simu_custom==np.array([]):
#        return Patch()
    simu_chk=simu_custom[0]+simu_custom[1]
    dff = df2.sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1]))
    if yaxis_column_name!=None and yaxis_column_name in df2:
        dff2 = dff[yaxis_column_name]
        title = '<b>{}</b>'.format(dff.Simu_Name.data)
        return selected_time_series(dff2,title,simu_chk)
    return Patch()




### 2D Seasonal Cycles




def create_seasonal(title, xcolumn_name,ycolumn_name,simucateg):
    fig  = Patch()
    data=[]
    for simcheck in simucateg:
        etau,simutype=coord_dict[simcheck]
        dset2=dset_sc.sel(nnetau=etau,SimuType=simutype).dropna(dim="NSimu",how="any",thresh=50)
        for ind in range(len(dset2[xcolumn_name])):
             data.append(go.Scatter(mode='lines+markers',name="",x=recollement(dset2[xcolumn_name][ind].data),y=recollement(dset2[ycolumn_name][ind].data),showlegend=False,opacity=0.1,hoverinfo='skip',marker={"size":3,"symbol":0,"color":"black"},line_color="black"))

    data.append(go.Scatter(mode='lines+markers+text',name="Sim",x=[], y=[],text=[],textposition=f'top right',textfont={'size':30,'color':['white','black','white','black','white','black','white','black','white','black','white','black','white']},marker=marker_dict4["LR0"],line=line_colors["LR0"]))
#    fig.layout.yaxis.type='linear' if yaxis_type == 'Linear' else 'log'
#    fig.layout.xaxis.type='linear' if xaxis_type == 'Linear' else 'log'
    fig["layout"]["annotations"][0].x=0
    fig["layout"]["annotations"][0].y=0.85
    fig["layout"]["annotations"][0].xanchor='left'
    fig["layout"]["annotations"][0].yanchor='bottom'
    fig["layout"]["annotations"][0].align='left'
    fig["layout"]["annotations"][0].text=title
    fig["layout"]["annotations"][0].font_size=16
    fig.layout.yaxis.title=ycolumn_name
    fig.layout.xaxis.title=xcolumn_name
    fig.data=data
    return fig

def selected_seasonal(dff1,dff2,title,simu_chk):
    fig  = Patch() 
    fig["data"][-1]['x']=recollement(dff1.data)
    fig["data"][-1]['y']=recollement(dff2.data)
    fig["data"][-1]['text']=["","F","M","A","M","J","J","A","S","O","N","D","J"]
    fig["data"][-1]["marker"]=marker_dict4[simu_chk]
    fig["data"][-1]["line"]=line_colors[simu_chk]
    fig["data"][-1]["name"]=title
    fig["layout"]["annotations"][0]["text"]=title 
    return fig

def empty_seasonal():
    fig=Patch()
    fig["data"]=[]
    fig.layout.yaxis.title=""
    fig.layout["annotations"][0].x=0.3
    fig.layout["annotations"][0].y=0.5
    fig.layout["annotations"][0].xanchor='auto'
    fig.layout["annotations"][0].yanchor='auto'
    fig.layout["annotations"][0].align='center'
    fig.layout["annotations"][0].text="NO  SEASONAL  CYCLE"
    fig.layout["annotations"][0].font_size=55
    return fig

@app.callback(
    Output('seasonal', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-xaxis-column', 'value'),
#    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
#    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-nnetau_0-check', 'value'),
    Input('crossfilter-nnetau_1-check','value'))
def update_all_seasonal(tab_selected,xaxis_column_name, yaxis_column_name, nnetau0,nnetau1):
    if tab_selected!='tab-2':
        return Patch()
    if xaxis_column_name!=None and Met_to_SC[xaxis_column_name] in dset_sc and yaxis_column_name!=None and Met_to_SC[yaxis_column_name] in dset_sc:
        title = '<b>{}</b>'.format("No Sim Selected")
        return create_seasonal(title, Met_to_SC[xaxis_column_name], Met_to_SC[yaxis_column_name],nnetau0+nnetau1)
    return empty_seasonal()

@app.callback(
    Output('seasonal', 'figure',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'))
def update_selected_seasonal(tab_selected,hoverData, xaxis_column_name,yaxis_column_name):
    if tab_selected!='tab-2':
        return Patch()
    simu_custom=hoverData['points'][0]['customdata']
#    if simu_custom==np.array([]):
#        return Patch()
    simu_chk=simu_custom[0]+simu_custom[1]
    dff = dset_sc.sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1]))
    if xaxis_column_name!=None and Met_to_SC[xaxis_column_name] in dset_sc and yaxis_column_name!=None and Met_to_SC[yaxis_column_name] in dset_sc:
        dff1 = dff[Met_to_SC[xaxis_column_name]]
        dff2 = dff[Met_to_SC[yaxis_column_name]]
        title = '<b>{}</b>'.format(dff.Simu_Name.data)
        return selected_seasonal(dff1,dff2,title,simu_chk)
    return Patch()

def recollement(dataset):
    return np.append(dataset,dataset[0])-dataset.mean()






### 2D & 3D Climatology Maps





@app.callback(
    Output('maps-x', 'figure',allow_duplicate=True),
    Output('input_maxx','value',allow_duplicate=True),
    Output('input_minx','value',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('EnsMaps','n_clicks'),
    Input('maps-flavor','value'),
    Input('input_maxx','value'),
    Input('input_minx','value'),
    Input('input_maxx','disabled'))
def update_maps_x(tab_selected, hoverData, xaxis_column_name,click,flavor,zup,zdown,auto):
    if tab_selected!='tab-3':
        figMap=go.Figure()
        figMap.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
        figMap.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="No Climatology Map",font_size=20)
        return figMap
    if auto==True:
        zup=None
        zdown=None
    if xaxis_column_name!=None and Met_to_Map[xaxis_column_name] in dset_map[Map_to_File[Met_to_Map[xaxis_column_name]]+"_LR"]:
        #title = '<b>{}</b>'.format("No Sim Selected")
        simu_custom=hoverData['points'][0]['customdata']
        if click==1:
            return create_meanmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name])
        if click==2:
            return create_stdmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name])
        if flavor=="Raw":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[xaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_raw_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name],zup,zdown)
        if flavor=="Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[xaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_ano_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name],zup,zdown)
        if flavor=="Standardised Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[xaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_anostd_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name],zup,zdown)
        #simu_custom=hoverData['points'][0]['customdata']
        #return create_map_raw_1col(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[xaxis_column_name])
    figMap=go.Figure()
    figMap.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    figMap.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                   xref='paper', yref='paper', showarrow=False, align='left',
                   text="No Climatology Map",font_size=20)
    return figMap

@app.callback(
    Output('maps-y', 'figure',allow_duplicate=True),
    Output('input_maxy','value',allow_duplicate=True),
    Output('input_miny','value',allow_duplicate=True),
    Input('tabs-2D','value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('EnsMaps','n_clicks'),
    Input('maps-flavor','value'),
    Input('input_maxy','value'),
    Input('input_miny','value'),
    Input('input_maxy','disabled'))
def update_maps_y(tab_selected, hoverData, yaxis_column_name,click,flavor,zup,zdown,auto):
    if tab_selected!='tab-3':
        figMap=go.Figure()
        figMap.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
        figMap.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="No Climatology Map",font_size=20)
        return figMap
    if auto==True:
        zup=None
        zdown=None
    if yaxis_column_name!=None and Met_to_Map[yaxis_column_name] in dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]+"_LR"]:
        simu_custom=hoverData['points'][0]['customdata']
        if click==1:
            return create_meanmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[yaxis_column_name])
        if click==2:
            return create_stdmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[yaxis_column_name])
        if flavor=="Raw":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_raw_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[yaxis_column_name],zup,zdown)
        if flavor=="Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_ano_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[yaxis_column_name],zup,zdown)
        if flavor=="Standardised Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_anostd_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[yaxis_column_name],zup,zdown)
    figMap=go.Figure()
    figMap.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    figMap.add_annotation(x=0, y=0.9, xanchor='left', yanchor='bottom',
                   xref='paper', yref='paper', showarrow=False, align='left',
                   text="No Climatology Map",font_size=20)
    return figMap

def create_map_raw_1col(simunum, simutype, nnetau, metric):
    dff = dset_map[Map_to_File[metric]+"_"+simutype].sel(NSimu=simunum,SimuType=simutype,nnetau=nnetau)
    fig=go.Figure(data=go.Heatmap(z=dff[metric]))
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig

def create_map_raw_enscol(simunum, simutype, nnetau, metric,zup,zdown):
    dff = dset_map[Map_to_File[metric]+"_"+simutype].sel(NSimu=simunum,SimuType=simutype,nnetau=nnetau)
    df_ens = dset_map[Map_to_File[metric]+"_"+simutype].sel(SimuType=simutype,nnetau=nnetau)[metric]
    if zup!=None and zdown!=None:
        zmaxi=zup
        zmini=zdown
    else:
        zmaxi=df_ens.max().item()
        zmini=df_ens.min().item()
    fig=go.Figure(data=go.Heatmap(z=dff[metric].data,zmin=zmini,zmax=zmaxi))
    fig.layout.autosize=True
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig,zmaxi,zmini

def create_map_ano_enscol(simunum, simutype, nnetau, metric,zup,zdown):
    dff = dset_map[Map_to_File[metric]+"_"+simutype].sel(NSimu=simunum,SimuType=simutype,nnetau=nnetau)
    df_ens=dset_map[Map_to_File[metric]+"_"+simutype].sel(SimuType=simutype,nnetau=nnetau)[metric]
    ens_min=(df_ens-df_ens.mean(dim="NSimu")).min().item()
    ens_max=(df_ens-df_ens.mean(dim="NSimu")).max().item()
    if zup!=None and zdown!=None:
        zmaxi=zup
        zmini=zdown
    else:
        zmaxi=max(-ens_min,ens_max)
        zmini=min(ens_min,-ens_max)
    mapping=(dff[metric]-df_ens.mean(dim="NSimu"))
    fig=go.Figure(data=go.Heatmap(z=mapping,zmin=zmini,zmax=zmaxi,colorscale="rdbu",reversescale=True))
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig,zmaxi,zmini

def create_map_anostd_enscol(simunum, simutype, nnetau, metric,zup,zdown):
    dff = dset_map[Map_to_File[metric]+"_"+simutype].sel(NSimu=simunum,SimuType=simutype,nnetau=nnetau)
    df_ens=dset_map[Map_to_File[metric]+"_"+simutype].sel(SimuType=simutype,nnetau=nnetau)[metric]
    ens_min=((df_ens-df_ens.mean(dim="NSimu"))/df_ens.std(dim="NSimu")).min().item()
    ens_max=((df_ens-df_ens.mean(dim="NSimu"))/df_ens.std(dim="NSimu")).max().item()
    if zup!=None and zdown!=None:
        zmaxi=zup
        zmini=zdown
    else:
        zmaxi=max(-ens_min,ens_max)
        zmini=min(ens_min,-ens_max)
    mapping=((dff[metric]-df_ens.mean(dim="NSimu"))/df_ens.std(dim="NSimu"))
    fig=go.Figure(data=go.Heatmap(z=mapping,zmin=zmini,zmax=zmaxi,colorscale="picnic"))
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig,zmaxi,zmini

def create_meanmap(simutype, nnetau, metric):
    df_ens=dset_map[Map_to_File[metric]+"_"+simutype].sel(SimuType=simutype,nnetau=nnetau)[metric]
    ens_min=((df_ens.mean(dim="NSimu"))).min().item()
    ens_max=((df_ens.mean(dim="NSimu"))).max().item()
    mapping=((df_ens.mean(dim="NSimu")))
    fig=go.Figure(data=go.Heatmap(z=mapping,zmin=ens_min,zmax=ens_max,colorscale="portland"))
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig,ens_min,ens_max

def create_stdmap(simutype, nnetau, metric):
    df_ens=dset_map[Map_to_File[metric]+"_"+simutype].sel(SimuType=simutype,nnetau=nnetau)[metric]
    ens_min=((df_ens.std(dim="NSimu"))).min().item()
    ens_max=((df_ens.std(dim="NSimu"))).max().item()
    mapping=((df_ens.std(dim="NSimu")))
    fig=go.Figure(data=go.Heatmap(z=mapping,zmid=0,colorscale="spectral"))
    fig.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    return fig,ens_min,ens_max


@app.callback(
    Output('EnsMaps','style'),
    Output('EnsMaps','children'),
    Input('EnsMaps','n_clicks'))
def button_flavour(click):
    if click%3==0:
        return {'backgroundColor':'#FFFFFF', 'padding':'0 1vw', 'width': '40%', 'display': 'inline', "verticalAlign": "top" ,'height':'36px', 'textAlign':'center'},"Show Mean or STD Map"
    if click%3==1:
        return {'backgroundColor':'#CCCCCC', 'padding':'0 1vw', 'width': '40%', 'display': 'inline', "verticalAlign": "top" ,'height':'36px', 'textAlign':'center'},"Ensemble Mean Map"
    else:
        return {'backgroundColor':'#FFCCCC', 'padding':'0 1vw', 'width': '40%', 'display': 'inline',  'height':'36px', 'textAlign':'center', "verticalAlign": "top"},"Ensemble STD Map"

@app.callback(
    Output('EnsMaps','n_clicks'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'))
def button_reset(xaxis_column_name,yaxis_column_name):
    return 0

@app.callback(
    Output('ScaleX','style'),
    Output('input_minx','disabled'),
    Output('input_maxx','disabled'),
    Output('ScaleX','children'),
    Output('input_minx','value',allow_duplicate=True),
    Output('input_maxx','value',allow_duplicate=True),
    Input('ScaleX','n_clicks'),
    State('ScaleX','style'))
def button_scale_x(click,style_button):
    if click%2==1:
        style_button["backgroundColor"]="#FFFFFF"
        return style_button,False,False,"AutoScale OFF",None,None
    else:
        style_button["backgroundColor"]="#CCCCCC"
        return style_button,True,True,"AutoScale ON",None,None


@app.callback(
    Output('ScaleY','style'),
    Output('input_miny','disabled'),
    Output('input_maxy','disabled'),
    Output('ScaleY','children'),
    Output('input_miny','value',allow_duplicate=True),
    Output('input_maxy','value',allow_duplicate=True),
    Input('ScaleY','n_clicks'),
    State('ScaleY','style'))
def button_scale_y(click,style_button):
    if click%2==1:
        style_button["backgroundColor"]="#FFFFFF"
        return style_button,False,False,"AutoScale OFF",None,None
    else:
        style_button["backgroundColor"]="#CCCCCC"
        return style_button,True,True,"AutoScale ON",None,None






@app.callback(
    Output('ScalePPM','style'),
    Output('input_minPPM','disabled'),
    Output('input_maxPPM','disabled'),
    Output('ScalePPM','children'),
    Output('input_minPPM','value',allow_duplicate=True),
    Output('input_maxPPM','value',allow_duplicate=True),
    Input('ScalePPM','n_clicks'),
    State('ScalePPM','style'))
def button_scale_z(click,style_button):
    if click%2==1:
        style_button["backgroundColor"]="#FFFFFF"
        return style_button,False,False,"AutoScale OFF",None,None
    else:
        style_button["backgroundColor"]="#CCCCCC"
        return style_button,True,True,"AutoScale ON",None,None


@app.callback(
    Output('EnsMapsPPM','n_clicks'),
    Input('crossparam-metric-column', 'value'))
def button_reset_PPM(metric_column_name):
    return 0

@app.callback(
    Output('EnsMapsPPM','style'),
    Output('EnsMapsPPM','children'),
    Input('EnsMapsPPM','n_clicks'))
def button_flavour_PPM(click):
    if click%3==0:
        return {'backgroundColor':'#FFFFFF', 'padding':'0 1vw', 'width': '40%', 'display': 'inline', "verticalAlign": "top" ,'height':'36px', 'textAlign':'center'},"Show Mean or STD Map"
    if click%3==1:
        return {'backgroundColor':'#CCCCCC', 'padding':'0 1vw', 'width': '40%', 'display': 'inline', "verticalAlign": "top" ,'height':'36px', 'textAlign':'center'},"Ensemble Mean Map"
    else:
        return {'backgroundColor':'#FFCCCC', 'padding':'0 1vw', 'width': '40%', 'display': 'inline',  'height':'36px', 'textAlign':'center', "verticalAlign": "top"},"Ensemble STD Map"

@app.callback(
    Output('maps-PPM', 'figure',allow_duplicate=True),
    Output('input_maxPPM','value',allow_duplicate=True),
    Output('input_minPPM','value',allow_duplicate=True),
    Input('crossparam-indicator-scatter3D', 'hoverData'),
    Input('crossparam-indicator-scatter2D', 'hoverData'),
    Input('crossparam-type','value'),
    Input('crossparam-metric-column', 'value'),
    Input('EnsMapsPPM','n_clicks'),
    Input('maps-PPM-flavor','value'),
    Input('input_maxPPM','value'),
    Input('input_minPPM','value'),
    Input('input_maxPPM','disabled'))
def update_maps_PPM(hoverData3D, hoverData2D, crossparam_type, metric_column_name,click,flavor,zup,zdown,auto):
    if "3D" in crossparam_type:
        hoverData=hoverData3D
    else:
        hoverData=hoverData2D
    if auto==True:
        zup=None
        zdown=None
    if metric_column_name!=None and Met_to_Map[metric_column_name] in dset_map[Map_to_File[Met_to_Map[metric_column_name]]+"_LR"]:
        simu_custom=hoverData['points'][0]['customdata']
        if click==1:
            return create_meanmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[metric_column_name])
        if click==2:
            return create_stdmap(simu_custom[0],int(simu_custom[1]),Met_to_Map[metric_column_name])
        if flavor=="Raw":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_raw_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[metric_column_name],zup,zdown)
        if flavor=="Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_ano_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[metric_column_name],zup,zdown)
        if flavor=="Standardised Anomaly":
            #title = '<b>{}</b>'.format(dset_map[Map_to_File[Met_to_Map[yaxis_column_name]]].sel(NSimu=int(simu_custom[2]),SimuType=simu_custom[0],nnetau=int(simu_custom[1])).Simu_Name.data)
            return create_map_anostd_enscol(int(simu_custom[2]),simu_custom[0],int(simu_custom[1]),Met_to_Map[metric_column_name],zup,zdown)
    figMap=go.Figure()
    figMap.update_layout(margin={'l': 0, 'b': 0, 'r': 0, 't': 0})
    figMap.add_annotation(x=0.3, y=0.5, xanchor='auto', yanchor='auto',
                   xref='paper', yref='paper', showarrow=False, align='center',
                   text="NO  CLIMATOLOGY  MAP",font_size=20)
    return figMap




if __name__ == '__main__':
    app.run_server(debug=True)
