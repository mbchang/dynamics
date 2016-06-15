
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

initialize_positions = function(num_obj, obj_radius, rand_pos_fn){
    var p0 = [];  // initial positions

    // set positions
    for (var i = 0; i < num_obj; i++) {

        // generate random initial positions by rejection sampling
        if (p0.length == 0) {  // assume that num_obj > 0
            p0.push(rand_pos_fn());
        } else {
            var proposed_pos = rand_pos_fn();
            // true if overlaps
            while ((function(){
                    for (var j = 0; j < p0.length; j++) {
                        if (euc_dist(proposed_pos, p0[j]) < 1.5*obj_radius) {
                            return true;
                        }
                    }
                    return false;
                })()){
                // keep trying until you get a match
                proposed_pos = rand_pos_fn();
            }
            p0.push(proposed_pos);
        }
    }
    return p0;
}

initialize_velocities = function(num_obj, max_v0) {
    var v0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial velocities b/w -max_v0 and max_v0 inclusive
        var vx = Math.floor(Math.random()*2*max_v0+1-max_v0)
        var vy = Math.floor(Math.random()*2*max_v0+1-max_v0)
        v0.push({x: vx, y: vy})
    }
    return v0;
}


// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = function(){
        this.euc_dist = euc_dist
        this.rand_pos = rand_pos
        this.initialize_positions = initialize_positions
        this.initialize_velocities = initialize_velocities
    };
}
