var router = __REQ_express.Router();
var request = __REQ_request;
var decryptRSA = require('../../../../../lib/rsa').decryptRSA;

const responseHandler = (req, res, expectedCode = 200) => {
    return function (error, response, body) {
        if (error) {
            __BRTC_ERROR_HANDLER.sendServerError(res, error);
        } else {
            if (response.statusCode === expectedCode) {
                res.json({ success: true });
            } else {
                __BRTC_ERROR_HANDLER.sendServerError(res, JSON.parse(body));
            }
        }
    };
};

var getDecryptedDatasource = function (req) {
    return Object.assign({}, req.body, {
        secretAccessKey: decryptRSA(req.body.secretAccessKey, req),
    });
};

var _executeInPermission = function (req, res, perm, task) {
    var permHandler = __BRTC_PERM_HELPER.checkPermission(req,
        [__BRTC_PERM_HELPER.PERMISSION_RESOURCE_TYPES.DATASOURCE], perm);
    permHandler.on('accept', task);
    permHandler.on('deny', function (permissions) {
        __BRTC_ERROR_HANDLER.sendNotAllowedError(res);
    });
    permHandler.on('fail', function (err) {
        __BRTC_ERROR_HANDLER.sendServerError(res, err);
    });
};

var convertConnection = function ({ accessKeyId, secretAccessKey, cloudType, _links }) {
    const hrefPrefix = '/api/core/v2/entity/connection/';

    const url = _links.self.href;
    return {
        accessKeyId,
        secretAccessKey: '',
        cloudType,
        connectionName: url.substring(url.indexOf(hrefPrefix) + hrefPrefix.length),
    };
};

var getConnection = function (req, res) {
    var task = function (permissions) {
        var connectionName = req.params.connectionName;
        var options = __BRTC_CORE_SERVER.createRequestOptions('GET', '/api/core/v2/entity/connection/' + connectionName);
        __BRTC_CORE_SERVER.setBearerToken(options, req.accessToken);
        request(options, function (error, response, body) {
            if (error) {
                return __BRTC_ERROR_HANDLER.sendServerError(res, error);
            } else {
                if (response.statusCode === 200) {
                    return res.json(convertConnection(JSON.parse(body)));
                } else {
                    // __BRTC_ERROR_HANDLER.sendServerError(res, JSON.parse(body));
                    // res.status(response.statusCode).send(response.body);
                    // 조회시 없는 datasource인 경우 에러(?)
                    if (body) {
                        return __BRTC_ERROR_HANDLER.sendServerError(res, JSON.parse(body));
                    }
                    return res.json(null);
                }
            }
        });
    };
    _executeInPermission(req, res, __BRTC_PERM_HELPER.PERMISSIONS.PERM_DATASOURCE_READ, task);
};

var listConnections = function (req, res) {
    var task = function (permissions) {
        var requestUrl = '/api/core/v2/entity/connection';
        if (req.query.cloudType) requestUrl += '/search/findByCloudType?cloudType=' + req.query.cloudType;

        var options = __BRTC_CORE_SERVER.createRequestOptions('GET', requestUrl);
        __BRTC_CORE_SERVER.setBearerToken(options, req.accessToken);
        request(options, function (error, response, body) {
            if (error) {
                __BRTC_ERROR_HANDLER.sendServerError(res, error);
            } else {
                if (response.statusCode === 200) {
                    const items = JSON.parse(body);
                    res.json(items._embedded.brtcCloudConnections.map(convertConnection));
                } else {
                    __BRTC_ERROR_HANDLER.sendServerError(res, JSON.parse(body));
                }
            }
        });
    };
    _executeInPermission(req, res, __BRTC_PERM_HELPER.PERMISSIONS.PERM_DATASOURCE_READ, task);
};

var createConnection = function (req, res) {
    var task = function (permissions) {
        var options = __BRTC_CORE_SERVER.createRequestOptions('POST', '/api/core/v2/entity/connection');
        __BRTC_CORE_SERVER.setBearerToken(options, req.accessToken);
        options.body = JSON.stringify(getDecryptedDatasource(req));
        request(options, responseHandler(req, res, 201));
    };
    _executeInPermission(req, res, __BRTC_PERM_HELPER.PERMISSIONS.PERM_DATASOURCE_UPDATE, task);
};

var updateConnection = createConnection;

var deleteConnection = function (req, res) {
    var task = function (permissions) {
        var connectionName = req.params.connectionName;
        var options = __BRTC_CORE_SERVER.createRequestOptions('DELETE', '/api/core/v2/entity/connection/' + connectionName);
        __BRTC_CORE_SERVER.setBearerToken(options, req.accessToken);
        request(options, responseHandler(req, res, 204));
    };
    _executeInPermission(req, res, __BRTC_PERM_HELPER.PERMISSIONS.PERM_DATASOURCE_DELETE, task);
};

router.get('/connection/:connectionName', getConnection);
router.get('/connection', listConnections);
router.post('/connection/:connectionName', createConnection);
router.post('/connection/:connectionName/update', updateConnection);
router.post('/connection/:connectionName/delete', deleteConnection);

module.exports = router;
