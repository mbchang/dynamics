// if (!_isBrowser) {
//     var jsonfile = require('jsonfile')
// }

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

initialize_angles = function(num_obj, max_a0) {
    var a0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial angles b/w -max_a0 and max_a0 inclusive
        var a = Math.random()*2*max_a0+1-max_a0
        a0.push(a)
    }
    return a0;
}

initialize_angle_velocities = function(num_obj, max_av0) {
    var av0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial angles b/w -max_a0 and max_a0 inclusive
        var av = Math.random()*2*max_av0+1-max_av0
        av0.push(av)
    }
    return av0;
}

initialize_masses = function(num_obj, possible_masses) {
    // TODO: this should be categorical!
    var masses = [];
    for (var i = 0; i < num_obj; i++) {

        // choose a random mass in the list of possible_masses
        var m = Math.floor(Math.random()*possible_masses.length)
        masses.push(possible_masses[m])
    }
    return masses;
}

// load_trajectory = function(file_path) {
//     var trajectory = jsonfile.readFileSync(file_path)
//     console.log(trajectory)
//     console.log(trajectory.length)
// }


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

// load_trajectory('balls.json')
