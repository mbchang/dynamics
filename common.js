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
                        if (euc_dist(proposed_pos, p0[j]) < 2.5*obj_radius) {
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

initialize_positions_variable_size = function(num_obj, sampled_sizes, rand_pos_fn){
    var p0 = [];  // initial positions

    // console.log(sampled_sizes)

    // set positions
    for (var i = 0; i < num_obj; i++) {
        let num_iters = 0
        let should_break = false

        // generate random initial positions by rejection sampling
        if (p0.length == 0) {  // assume that num_obj > 0
            p0.push(rand_pos_fn());
        } else {
            var proposed_pos = rand_pos_fn();
            // true if overlaps
            while ((function(){
                    for (var j = 0; j < p0.length; j++) {
                        let other_size = sampled_sizes[j]  // this is the raw size, sizemul already incorporated. For an obstacle, let it be the diagonal from the center?
                        let this_size = sampled_sizes[i]
                        let min_distance = other_size + this_size
                        // console.log('min_dist',i,j,min_distance)

                        if (euc_dist(proposed_pos, p0[j]) < 1.25*min_distance) {
                            // num_iters ++
                            // if (num_iters > 100) {
                            //     should_break = true
                            //     break
                            // }
                            return true;
                        }
                    }
                    // if (should_break) {
                    //     break
                    // }

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

initialize_hv = function(num_obj) {
    var a0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial angles b/w -max_a0 and max_a0 inclusive
        var a = Math.floor(Math.random()*2)*Math.PI/2
        a0.push(a)
    }
    return a0;
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

initialize_sizes = function(num_obj, possible_sizes) {
    // TODO: this should be categorical!
    var sizes = [];
    for (var i = 0; i < num_obj; i++) {

        // choose a random mass in the list of possible_masses
        var s = Math.floor(Math.random()*possible_sizes.length)
        sizes.push(possible_sizes[s])
    }
    return sizes;
}

// assume trajectories ordered from bottom to top
is_stable_trajectory = function(trajectories) {
    // not stable if top block's y position is different it's original y position by a factor of a block length
    // but what if it is horizontal?
    // how about x position?
    // or you can just do Euclidean distance
    let top = trajectories[trajectories.length-1]
    let initial = {x: top[0].position.x, y: top[0].position.y}
    let final = {x: top[top.length-1].position.x, y: top[top.length-1].position.y}
    let dist = euc_dist(final, initial)
    return dist
}

// objects: array of bodies
// center of mass
// origin: demo.cx
// assume that bodies are ordered bottom to top
// center_of_mass = function(bodies) {
//     let com
//     if (bodies.length == 1) {
//         com = bodies[0].position.x
//     } else {
//         // jth iteration: (prev)*((j-1)/j) + curr/j
//         let above = bodies.slice(0,-1)  // all elements before last element
//         let body = bodies[bodies.length-1]  // last element
//         let len = bodies.length
//         com = center_of_mass(above)*(len-1)/len + body.mass*body.position.x/len
//     }
//     // console.log(com)
//     return com
// }

// objects: array of bodies
// center of mass
// origin: demo.cx
// assume that bodies are ordered bottom to top
center_of_mass2 = function(bodies) {
    if (bodies.length == 1) {
        return bodies[0].position.x
    } else {
        let total_mass = 0
        let sum = 0
        let coms = []
        for (let i=0; i < bodies.length; i++) {
            if (i == 0) {
                coms.push(bodies[i].position.x)
            } else {
                // jth iteration: (prev)*((j-1)/j) + curr/j
                coms.push(coms[i-1]*(i)/(i+1) + bodies[i].mass*bodies[i].position.x/(i+1))  // zero indexed
            }
            total_mass += bodies[i].mass
            sum += bodies[i].mass*bodies[i].position.x
        }
        // return sum/total_mass  // you could also just return this if you want the last one
        return coms
    }
}

center_of_mass = function(bodies) {
    if (bodies.length == 1) {
        return bodies[0].position.x
    } else {
        let total_mass = 0
        let sum = 0
        let coms = []
        for (let num_bodies=0; num_bodies < bodies.length; num_bodies++) {  // go from top most block down
            let curr_body = bodies[bodies.length-num_bodies-1]
            if (num_bodies == 0) {
                coms.push(curr_body.position.x)
            } else {
                // jth iteration: (prev)*((j-1)/j) + curr/j
                coms.push(coms[num_bodies-1]*(num_bodies)/(num_bodies+1) + curr_body.mass*curr_body.position.x/(num_bodies+1))  // zero indexed
            }
            total_mass += curr_body.mass
            sum += curr_body.mass*curr_body.position.x
        }
        //return sum/total_mass  // you could also just return this if you want the last one
        return coms
    }
}



// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = function(){
        this.euc_dist = euc_dist
        this.rand_pos = rand_pos
        this.initialize_positions = initialize_positions
        this.initialize_velocities = initialize_velocities
        this.initialize_hv = initialize_hv
        this.initialize_positions_variable_size = initialize_positions_variable_size
    };
}

// load_trajectory('balls.json')


