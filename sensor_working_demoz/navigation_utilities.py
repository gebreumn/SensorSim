# -*- coding: utf-8 -*-
"""
                 navigation_utilities.py
==============================================================

This script defines navigation utility functions used in mechanizing
aided inertial navigation systems.  The functions included are:

1)


    @Author:        Demoz Gebre-Egziabher 
    Created:        July 3, 2013   
    Last Modified:  July 3, 2013    
    Copywrite 2013  Demoz Gebre-Egziabher
    License: BSD, see bsd.txt for details 
=============================================================

"""
#   Import relevant libraries

import numpy
import math

#   Define Contstants

f = 1.0/298.257223563      #   WGS-84 Flattening.
e = math.sqrt(f*(2.0 - f)) #   Eccentricity.
omega_ie = 7.292115e-5     #   WGS-84 Earth rate (rad/s).
R_0 = 6378137              #   WGS-84 equatorial radius (m).                            
R_P = R_0*(1.0 - f)        #   Polar radius (m).
mu_E = 3.986004418e14      #   WGS-84 Earth's gravitational parameter


#                           eul2Cbn(eul,sequence)
#   eul2Cbn converts an arbitrary euler angle sequence (specified) to a
#   direction math.comath.sine matrix Cbn which maps vectors from the b frame to the
#   n frame.  Note that the euler angles are the n to b rotations (consistent
#   with aerospace conventions) and sequcne is an 1 x 3 array of numbers 
#   indicating the sequence of rotations

def eul2Cbn(eul):   # DCM Calculation

    C1 = numpy.array([[1.0,0.0,0.0],[0.0, math.cos(eul[0]), math.sin(eul[0])],[0.0, -math.sin(eul[0]),math.cos(eul[0])]])
    C2 = numpy.array([[math.cos(eul[1]), 0.0, -math.sin(eul[1])],[0.0, 1.0, 0.0],[math.sin(eul[1]), 0.0, math.cos(eul[1])]])
    C3 = numpy.array([[math.cos(eul[2]),math.sin(eul[2]),0.0],[-math.sin(eul[2]),math.cos(eul[2]),0.0],[0.0,0.0,1.0]])
    Cnb = numpy.dot(C1,numpy.dot(C2,C3))
    return Cnb.transpose()

def Cbn2eul(Cbn):
    
    Cnb = Cbn.transpose()    
    eul = numpy.array([0.0, 0.0, 0.0])
    eul[0] = math.atan2(Cnb[1][2],Cnb[2][2])
    eul[1] = -math.asin(Cnb[0][2])
    eul[2] = math.atan2(Cnb[0][1],Cnb[0][0])
    return eul
    
def earthrate(L):

    omega_n_ie = omega_ie*numpy.array([math.cos(L), 0.0, -math.sin(L)])
    return omega_n_ie

def earthrad(L):
    
    r_Earth = numpy.array([0.0, 0.0])
    k = math.sqrt(1.0 - math.pow(e*math.sin(L),2))
    r_Earth[0] = R_0*(1.0 - math.pow(e,2))/math.pow(k,3)
    r_Earth[1] = R_0/k
    return r_Earth

def navrate(v,p):
    
    r_Earth = earthrad(p[0])
    nav_rate = numpy.array([0.0, 0.0, 0.0])
    nav_rate[0] = v[1]/(r_Earth[1] + p[2])
    nav_rate[1] = -v[0]/(r_Earth[0] + p[2])
    nav_rate[2] = -v[1]*math.tan(p[0])/(r_Earth[1] + p[2])
    return nav_rate

def glocal(L,h):    
    
    g_0 = g_0 = (9.7803253359/(math.sqrt(1 - math.pow(e*math.sin(L),2))))*( 1 + 0.001931853*math.pow(math.sin(L),2))
    k = 1.0 - (2.0*h/R_0)*(1.0 + f + (math.pow(omega_ie*R_0,2))*(R_P/mu_E)) + 3.0*math.pow((h/R_0),2)
    g_n = numpy.array([0.0, 0.0, k*g_0])    
    return g_n
    
def skew(v_in):
    
    S = numpy.array([[0.0, -v_in[2], v_in[1]],[v_in[2], 0.0, -v_in[0]],[-v_in[1], v_in[0], 0.0]])
    return S
    
def wgsxyz2lla(p_e):
     
    x = p_e[0]
    y = p_e[1]
    z = p_e[2]

    #   Calculate longitude

    lon = math.atan2(y,x)*(180.0/math.pi)
    
    #   Start computing intermediate variables needed to compute altitude
    
    p = math.sqrt(x*x + y*y)
    E = math.sqrt(R_0*R_0 - R_P*R_P)
    F = 54.0*math.pow(R_P*z,2)
    G = math.pow(p,2) + (1.0 - math.pow(e,2))*math.pow(z,2) - math.pow(e*E,2)
    c = math.pow(e,4)*F*math.pow(p,2)/math.pow(G,3)
    s = math.pow((1.0 + c + math.sqrt(c*c + 2.0*c)),(1.0/3.0))
    P = (F/(3.0*G*G))/(math.pow((s + (1.0/s) + 1.0),2))
    Q = math.sqrt(1.0 + 2.0*math.pow(e,4)*P)
    k_1 = -P*math.pow(e,2)*p/(1.0 + Q)
    k_2 = 0.5*math.pow(R_0,2)*(1.0 + 1.0/Q)
    k_3 = -P*(1 - math.pow(e,2))*math.pow(z,2)/(Q*(1.0 + Q))
    k_4 = -0.5*P*math.pow(p,2)
    r_0 = k_1 + math.sqrt(k_2 + k_3 + k_4)
    k_5 = p - math.pow(e,2)*r_0
    U = math.sqrt(math.pow(k_5,2) + math.pow(z,2))
    V = math.sqrt(math.pow(k_5,2) + (1.0 - math.pow(e,2))*math.pow(z,2))
    alt = U*(1.0 - (math.pow(R_P,2)/(R_0*V)))

    #   Compute additional values required for computing
    #   latitude

    z_0 = (math.pow(R_P,2)*z)/(R_0*V)
    e_p = (R_0/R_P)*e
    lat = math.atan((z + z_0*math.pow(e_p,2))/p)*(180/math.pi)
    
    p_n = numpy.array([lat, lon, alt])
    return p_n
    
def wgslla2xyz(p_n):

    lat = p_n[0]
    lon = p_n[1]
    alt = p_n[2]
    
    #   Compute East-West Radius of curvature at current position

    R_E = R_0/(math.sqrt(1.0 - math.pow(e*math.sin(lat),2)))

    #   Compute ECEF coordinates

    p_e = numpy.array([0.0, 0.0, 0.0])
    p_e[0] = (R_E + alt)*math.cos(lat)*math.cos(lon)
    p_e[1] = (R_E + alt)*math.cos(lat)*math.sin(lon)
    p_e[2] = ((1.0 - math.pow(e,2))*R_E + alt)*math.sin(lat)

    return p_e    