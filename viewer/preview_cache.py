"""Bounded exact-request cache for local previews, never benchmark inference.

A compute lock prevents simultaneous preview requests from oversubscribing one
model/GPU. Responses are copied, and failures are never retained.
"""
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
import json


class PreviewCache:
    def __init__(self,entries=8,bytes_limit=32*1024*1024):
        self.entries=entries;self.bytes_limit=bytes_limit;self.items=OrderedDict();self.size=0;self.lock=RLock()

    def get_or_compute(self,key,compute):
        with self.lock:
            if key in self.items:
                value,size=self.items.pop(key);self.items[key]=(value,size)
                result=deepcopy(value);result['cache_hit']=True;return result
            value=compute();size=len(json.dumps(value,separators=(',',':')).encode())
            if size<=self.bytes_limit:
                self.items[key]=(deepcopy(value),size);self.size+=size
                while len(self.items)>self.entries or self.size>self.bytes_limit:
                    _,(_,old)=self.items.popitem(last=False);self.size-=old
            result=deepcopy(value);result['cache_hit']=False;return result

    def clear(self):
        with self.lock:self.items.clear();self.size=0
