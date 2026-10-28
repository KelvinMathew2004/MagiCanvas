"""Bounded request-owned asset discovery; no cross-request or geometry cache."""
from contextvars import ContextVar
from functools import wraps
import os

_STATE=ContextVar('magicanvas_asset_lookup',default=None)


def asset_lookup_state():
    return _STATE.get()


def asset_lookup_request(function):
    @wraps(function)
    def call(*args,**kwargs):
        enabled=os.environ.get('MAGICANVAS_ASSET_LOOKUP_CACHE','1')!='0'
        state=dict(roots={},calls=0,hits=0) if enabled else None
        token=_STATE.set(state)
        try:
            output,report=function(*args,**kwargs)
            report['asset_lookup_cache']=dict(enabled=enabled,calls=state['calls'] if state else 0,
                hits=state['hits'] if state else 0,roots_resolved=len(state['roots']) if state else 0)
            return output,report
        finally:
            _STATE.reset(token)
    return call
