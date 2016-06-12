var Matter = require('matter-js')

copy = function(thing) {
    if (typeof thing == 'object') {
        return Matter.Common.clone(thing, true)
    } else {
        return thing;
    }
}

// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = {
        copy
    };
}
