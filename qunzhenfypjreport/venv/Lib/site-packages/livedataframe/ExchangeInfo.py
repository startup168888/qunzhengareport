import json
import requests

class ExchangeInfo():
    ip = 'api.livedataframe.com'

    @classmethod
    def list_exchanges(cls):
#         r = requests.get('http://app.livedataframe.com/exchange')
        response = requests.get('http://%s/api/v1/exchange' %(cls.ip))
        return(response.json()['exchanges'])

    @classmethod
    def list_symbols_for(cls, exchange):
        response = requests.get('http://%s/api/v1/active_symbols?exchange=%s' %(cls.ip, exchange.lower()))
        return(response.json()['symbols'])
