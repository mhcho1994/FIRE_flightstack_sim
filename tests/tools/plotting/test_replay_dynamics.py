"""Coordinate and sampling checks for native replay dynamics diagnostics."""
import numpy as np

from tools.plotting.analyze_replay_dynamics import interpolate, quaternion_matrix, attitude_degrees


def test_ned_heading_and_quaternion_sign():
    # Positive 90-degree heading rotates body forward (North) to East.
    q=np.array([[2**-.5,0,0,2**-.5]])
    r=quaternion_matrix(q)
    np.testing.assert_allclose(r[0] @ [1,0,0], [0,1,0], atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(-q),r)
    np.testing.assert_allclose(attitude_degrees(q), [[0,0,90]],atol=1e-12)


def test_tilted_specific_force_and_gravity():
    # A nose-down 30-degree vehicle with T/m=g/cos(30) has zero vertical
    # acceleration and positive North acceleration g*tan(30), before drag.
    theta=np.deg2rad(-30)
    q=np.array([[np.cos(theta/2),0,np.sin(theta/2),0]])
    r=quaternion_matrix(q)[0]
    g=9.80665
    accel=r @ [0,0,-g/np.cos(theta)] + [0,0,g]
    np.testing.assert_allclose(accel, [g*np.tan(-theta),0,0],atol=1e-12)


def test_resampling_rejects_extrapolation_gaps_and_invalid_samples():
    result=interpolate([0,.1,2,2.1], [0,1,2,3], [-1,.05,1,2.05,3])[:,0]
    np.testing.assert_allclose(result[[1,3]],[.5,2.5])
    assert np.isnan(result[[0,2,4]]).all()
    assert np.isnan(interpolate([0,.1,.2], [0,np.nan,2], [.05,.15])).all()


def test_discrete_state_uses_past_and_duplicate_latest_publication():
    result=interpolate([0,.1,.1,.2], [3,4,5,6], [.05,.1,.15],hold=True)[:,0]
    np.testing.assert_array_equal(result,[3,5,5])


def test_exact_publication_remains_valid_before_invalid_next_sample():
    result=interpolate([0,.1,.2],[1,np.nan,3],[0,.05,.2])[:,0]
    assert result[0] == 1 and result[2] == 3
    assert np.isnan(result[1])
