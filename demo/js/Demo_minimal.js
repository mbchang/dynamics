/**
* The Matter.js demo page controller and example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @class Demo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.test(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');

    var Demo = {};
    Matter.Demo = Demo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile')
        var CircularJSON = require('circular-json')
        var assert = require('assert')
        var utils = require('../../utils')
        // var sleep = require('sleep')
        var PImage = require('pureimage');
        var fs = require('fs');
        require('./Examples')
        module.exports = Demo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Runner = Matter.Runner,
        Render = Matter.Render;
        Axes = Matter.Axes;

    // Create the engine
    Demo.run = function(json_data, opt) {

        //TODO: note that here you should load the demo engine with the json file

        // load the config file here.
        // console.log(json_data)
        let data = json_data.trajectories
        let config = json_data.config
        // console.log(config)

        // let data = json_data

        var demo = {}
        demo.offset = 5;  // world offset
        demo.config = {}
        demo.config.cx = 400;
        demo.config.cy = 300;
        demo.config.masses = [1, 5, 25]
        demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        demo.config.sizes = [2/3, 1, 3/2]  // multiples
        demo.config.drastic_sizes = [1/2, 1, 2]  // multiples  NOTE THAT WE HAVE THREE VALUES NOW!
        // demo.config.drastic_sizes = [1/2, 2]  // multiples  NOTE THAT WE HAVE THREE VALUES NOW!
        demo.config.object_base_size = {'ball': 60, 'obstacle': 80, 'block': 20 }  // radius of ball, side of square obstacle, long side of block
        demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        demo.config.g = 0 // default? [0,1] Or should we make this a list? The index of the one hot. 0 is no, 1 is yes
        demo.config.f = 0 // default? [0,1]
        demo.config.p = 0 // default? [0,1,2]
        demo.config.max_velocity = 60

        demo.cx = demo.config.cx;
        demo.cy = demo.config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        demo.engine = Engine.create()
        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                    max: { x: demo.width, y: demo.height }}


        // here let's put a isBrowser condition
        if (_isBrowser) {  // do everything normally.
            demo.runner = Engine.run(demo.engine)
            demo.runner.isFixed = true
            demo.container = document.getElementById('canvas-container');
            demo.render = Render.create({element: demo.container, engine: demo.engine, 
                                        hasBounds: true, options:{height:demo.height, width:demo.width}})
            Render.run(demo.render)
        } else {
            // run the engine
            demo.runner = Runner.create()
            demo.runner.isFixed = true
            var pcanvas = PImage.make(demo.width, demo.height);  // 693
            pcanvas.style = {}  
            console.log(pcanvas)
            demo.render = Render.create({
                element: 17, // dummy
                canvas: pcanvas,
                engine: demo.engine,
            })
            

            // Events.on(demo.render, 'afterRender', function(e) {
            //     var img = demo.render.canvas.toDataURL("image"+e.timestamp+".png")
            //     var data = img.replace(/^data:image\/\w+;base64,/, "");
            //     // var buf = new Buffer(data, 'base64');
            //     // fs.writeFile('image.png', buf);
            //     // document.write('<img src="'+img+'"/>');  // how do I save this?
            //     // PImage.encodePNG(img, fs.createWriteStream('out.png'), function(err) {
            //     // console.log("wrote out the png file to out.png");
            //     // });
            // });

            demo.render.hasBounds = true
            demo.render.options.height = demo.height
            demo.render.options.width = demo.width
            demo.render.canvas.height = demo.height
            demo.render.canvas.width = demo.width
        }


        if (demo.render) {
            var renderOptions = demo.render.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = true;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.background = '#fff';
        }

        var mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}// TODO eventually call Example[config.env].mass_colors

        // Ok, now let's manually update
        if (_isBrowser) {
            Runner.stop(demo.runner) // seems like this is causing the problem!
        }

        var trajectories = data[opt.ex]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length
        config.trajectories = trajectories

        Example[config.env](demo, config)  // here you have to assign balls initial positions according to the initial timestep of trajectories.


        if (config.env == 'tower') {
            var stability_threshold = 5
        }


        // if (config.env=='tower') {
        //     var s = 2  // if I set s to < 2 then I get very weird behavior
        // } else {
        //     var s = 0
        // }
        let s = 0
        // let gt_angles = [0, 0, 0.00013780457084067, 0.00019076203170698]

        function f() {
            console.log( 's =', s );
            var entities = Composite.allBodies(demo.engine.world)
                .filter(function(elem) {
                            return elem.label === 'Entity';
                        })
            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
                var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                if (s < config.num_past) {
                    body.render.strokeStyle = '#FFA500'// orange #551A8B is purple
                } else {
                    body.render.strokeStyle = '#551A8B'// orange #551A8B is purple
                }
                body.render.lineWidth = 5

                // set velocity
                // body.velocity = {x:0, y: 0}
                Body.setVelocity(body, {x: 0, y: 0})

                Body.setPosition(body, trajectories[id][s].position)

                if (id == 1) {
                    console.log('HOHOHO',s)
                    // Body.setAngle(body, 0)  // this makes the bottom block move!  interesting. It seems that the angle is making it do this!
                    // if (s < -1) {
                    //     Body.setAngle(body, 0)  // this makes the bottom block move!
                    // } else {
                        // Body.setAngularVelocity(body, 0)  // seems like we need to do this?
                        console.log('before angle', body.angle)
                        // let delta = trajectories[id][s].angle - body.angle
                        Body.setAngularVelocity(body, 0)  // seems like we need to do this?
                        // body.angle = trajectories[id][s].angle
                        Body.setAngle(body, trajectories[id][s].angle)  // this makes the bottom block move! it seems like setAngle doesn't work, but directly assining the angle does the trick?  
                        // Body.setAngle(body, body.angle)  // this makes the bottom block move! it seems like setAngle doesn't work, but directly assining the angle does the trick?  
                        // body.angle = trajectories[id][s].angle
                        // Body.setAngularVelocity(body, trajectories[id][s].angularVelocity)  // seems like we need to do this?
                        // Body.setAngularVelocity(body, 0)  // seems like we need to do this?
                        // body.angularVelocity = 0
                        // body.angle = 6.188708782196//trajectories[id][s].angle

                        // for tower
                        // for (let jk = 0; jk < body.parts.length; jk ++){
                        //     Axes.rotate(body.parts[jk].axes, -delta)  // let's show axes though
                        // }
                    // }
                    console.log('vel',body.velocity)
                    console.log('pos',body.position)
                    console.log('ang',body.angle)
                    console.log('av',body.angularVelocity)
                    console.log('tang',trajectories[id][s].angle)
                    console.log('LLLLLLLL')
                } else {
                    // if (s==0 || s == 1) {
                    //     Body.setAngle(body, 0)  // this makes the bottom block move!
                    // } else if (s == 2) {
                    //     Body.setAngle(body, 0.00013780457084067)  // this makes the bottom block move!
                    // } else if (s == 3) {
                    //     Body.setAngle(body, 0.00019076203170698)
                    // } else {
                        // body.angle = trajectories[id][s].angle
                        Body.setAngularVelocity(body, 0)  // seems like we need to do this?
                        Body.setAngle(body, trajectories[id][s].angle)  // this makes the bottom block move!
                        // body.angle = trajectories[id][s].angle
                        // body.angularVelocity = 0
                        // Body.setAngularVelocity(body, trajectories[id][s].angularVelocity)  // seems like we need to do this?

                    // }
                }


                // Body.setVelocity(body, {x: 0, y: 0})

                // Body.setPosition(body, trajectories[id][s].position)
               // if (s==0 || s == 1) {
               //      Body.setAngle(body, 0)  // this makes the bottom block move!
               //  } else if (s == 2) {
               //      // Body.setAngle(body, 0.00013780457084067)  // this makes the bottom block move!
               //      Body.setAngle(body, 0)  // this makes the bottom block move!

               //  } else if (s == 3) {
               //      // Body.setAngle(body, 0.00019076203170698)
               //      Body.setAngle(body, 0)  // this makes the bottom block move!

               //  } else {
               //      Body.setAngle(body, 0)  // this makes the bottom block move!
               //      // Body.setAngle(body, trajectories[id][s].angle)  // this makes the bottom block move!
               //  }






            }

            // stack upwards
            // for (let id = 0; id < num_obj; id ++) {
            //     // if (s==1) {
            //     let body_opts = {label: "Entity", 
            //                      restitution: 0, 
            //                      mass: trajectories[id][1].mass, 
            //                      objtype: trajectories[id][1].objtype,
            //                      sizemul: trajectories[id][1].sizemul, 
            //                      friction: 1,
            //                      collisionFilter: {group:Body.nextGroup(true)} // remove collision constraints
            //                  }
            //     let pos = trajectories[id][1].position          
            //     var block = Bodies.rectangle(pos.x, pos.y, 
            //                                  demo.config.object_base_size.block*body_opts.sizemul, 
            //                                  3*demo.config.object_base_size.block*body_opts.sizemul, 
            //                                  body_opts)
            //     Body.setAngle(block, trajectories[id][1].angle)
            //     Body.setVelocity(block, { x: 0, y: 0 })
            //     console.log(block.velocity)


            //     // set the position here
            //     if (id < config.num_past) {
            //         block.render.strokeStyle = '#FFA500'// orange #551A8B is purple
            //     } else {
            //         block.render.strokeStyle = '#551A8B'// orange #551A8B is purple
            //     }
            //     block.render.lineWidth = 5

            //     if (id==1) {
            //         block.render.fillStyle = 'black'
            //     } else {
            //         block.render.fillStyle = 'black'//self.mass_colors[trajectories[id][1].mass]//'#4ECDC4'
            //     }
            //     // block.render.fillStyle = self.mass_colors[trajectories[s][1].mass]//'#4ECDC4'


            //     block.render.strokeStyle = '#FFA500'// orange
            //     block.render.lineWidth = 5

            //     console.log('add to world', s)
            //     World.add(demo.engine.world, block)
            //     // }
            // }






            // can't clear the world border



            // Runner.tick(demo.runner, demo.engine);  // ok if I'm just rendering I DON'T NEED THIS! ACTUALY I DON'T NEED THIS AT ALL!

            if (config.env == 'tower') {
                if (s == 59) {
                    console.log('euc dist', s, is_stable_trajectory(trajectories))
                    console.log('stable?', s, is_stable_trajectory(trajectories) < stability_threshold)
                } else if (s == 119) {
                    console.log('euc dist', s, is_stable_trajectory(trajectories))
                    console.log('stable?', s, is_stable_trajectory(trajectories) < stability_threshold)
                } 
            }


            if (!_isBrowser) {
                demo.render.context.globalAlpha = 0.5
                demo.render.context.fillStyle = 'white'
                // demo.render.context.fillStyle = "rgba(255, 255, 255, 1.0)";
                demo.render.context.fillRect(0,0,demo.width,demo.height)
                Render.world(demo.render)
                let filename = opt.out_folder + '/' + opt.batch_name + '_ex' + opt.ex + '_step' + s +'.png'

                // let filename = 'out'+s+'_'+s+'.png'  // TODO! rename
                PImage.encodePNG(demo.render.canvas, fs.createWriteStream(filename), function(err) {
                    console.log("wrote out the png file to "+filename);
                });

            }

            // var entities = Composite.allBodies(demo.engine.world)
            //     .filter(function(elem) {
            //                 return elem.label === 'Entity';
            //             })
            // var entity_ids = entities.map(function(elem) {
            //                     return elem.id});

            // for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
            //     var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
            //     Composite.remove(demo.engine.world, body, true)
            // }


            s++;
            if( s < num_steps ){
                if (_isBrowser) {
                    setTimeout( f, 100 );
                } else {
                    setTimeout( f, 0 );
                }
            }
        }
        setInterval(function(){
          console.log('test');
        }, 1000);
        f();

        if (config.env == 'tower') {
            console.log('Fraction unstable',fraction_stable(trajectories,1))
            return [is_stable_trajectory(trajectories) < stability_threshold, is_stable_trajectory(trajectories)]  // true if unstable
        }
    };


    Demo.process_cmd_options = function() {
        const optionator = require('optionator')({
            options: [{
                    option: 'help',
                    alias: 'h',
                    type: 'Boolean',
                    description: 'displays help',
                }, {
                    option: 'exp',
                    alias: 'e',
                    type: 'String',
                    description: 'experiment folder',
                    required: true
                }]
        });

        // process invalid optiosn
        try {
            optionator.parseArgv(process.argv);
        } catch(e) {
            console.log(optionator.generateHelp());
            console.log(e.message)
            process.exit(1)
        }

        const cmd_options = optionator.parseArgv(process.argv);
        if (cmd_options.help) console.log(optionator.generateHelp());
        return cmd_options;
    };

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        // console.log('a')
        window.loadFile = function loadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                // let options = {out_folder: out_folder, ex: 0, batch_name: batch_name}
                Demo.run(window.CircularJSON.parse(fr.result), {ex:0})
            }
            fr.readAsText(file)
        }
    } else {
        // here load the json file here
        const cmd_options = Demo.process_cmd_options();
        console.log('processed command options', cmd_options)
        let experiment_folder = cmd_options.exp  // this is the folder that ends with predictions
        // let experiment_folder = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/opmjlogs/balls_n8_t60_ex50000_rd__balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj/balls_n8_t60_ex50000_rdpredictions'
        let jsons = fs.readdirSync(experiment_folder)

        for (let j=0; j < jsons.length; j++) {
            let jf = jsons[j]
            let loaded_json = jsonfile.readFileSync(experiment_folder + '/' + jf)
            let batch_name = jf.slice(0, -1*'.json'.length)
            let out_folder = experiment_folder + '/../visual/' + batch_name

            let stability_dists = {}

            if (loaded_json.config.env=='tower') {
                let num_stable = 0
                let num_unstable = 0
                for (let b=0; b < 50; b ++) {
                    let options = {out_folder: out_folder, ex: b, batch_name: batch_name}
                    console.log(batch_name)
                    let is_stable_data = Demo.run(loaded_json, options)
                    let is_stable = is_stable_data[0]
                    let euc_dist_stable = is_stable_data[1]
                    console.log('euc dist: ' + euc_dist_stable)
                    stability_dists[batch_name+'_ex'+b] = euc_dist_stable;
                    console.log('>>>>>>>>>>>>>>>>>>>>>>>>>')
                    if (is_stable) {
                        num_stable ++;
                    } else {
                        num_unstable ++;
                    }
                }
                console.log('############################')
                console.log(num_stable + ' stable ' + num_unstable + ' unstable for ' + out_folder)
                console.log('############################')
                console.log(stability_dists)
                jsonfile.writeFileSync(out_folder+'/stability_stats.json', top_block_deviation=stability_dists)
                console.log('Wrote to ' + out_folder+'/stability_stats.json')
            } else {
                let options = {out_folder: out_folder, ex: 0, batch_name: batch_name}
                console.log(batch_name)
                Demo.run(loaded_json, options)
                console.log('>>>>>>>>>>>>>>>>>>>>>>>>>')
            }
        }
        // console.log(jsons)

        // instead of a for loop I should use a call back.

        // but if it is asynchronous may be it will be faster?   




    }
})();


// TODO pass in the example in the batch as well as the experiment folder. You can create a image folder if you want 



