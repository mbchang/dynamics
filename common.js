
// common functions (I could pass these in later)
euc_dist = function(p1, p2) {
    var x2 = Math.pow(p1.x - p2.x, 2),
        y2 = Math.pow(p1.y - p2.y, 2);
    return Math.sqrt(x2 + y2);
},

rand_pos = function(x_bounds, y_bounds) {

    var xrange = x_bounds.hi - x_bounds.lo,
        yrange = y_bounds.hi - y_bounds.lo;
    var px = Math.floor((Math.random()*(xrange))) + x_bounds.lo,
        py = Math.floor((Math.random()*(yrange))) + y_bounds.lo;
    return {x: px, y: py};
}


// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = function(){
        this.euc_dist = euc_dist
        this.rand_pos = rand_pos
    };
}
