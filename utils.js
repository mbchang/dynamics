var Matter = require('matter-js')

copy = function(thing) {
    if (typeof thing == 'object') {
        return Matter.Common.clone(thing, true)
    } else {
        return thing;
    }
};

// chunk n into chunks of size k or less (last chunk will have size <= k)
chunk = function(n, k) {
    let chunks = [];
    while (n > k) {
        chunks.push(k);
        n -= k;
    }
    // at this point n <= k
    chunks.push(n)
    return chunks;
};

// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = {
        copy, chunk
    };
}
