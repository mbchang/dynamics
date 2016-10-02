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
        var path = require('path')
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

        console.log(opt)

        var trajectories = data[opt.ex]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length
        config.trajectories = trajectories

        Example[config.env](demo, config)  // here you have to assign balls initial positions according to the initial timestep of trajectories.


        if (config.env == 'tower') {
            var stability_threshold = 5
        }

        let s = 0

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
                Body.setVelocity(body, {x: 0, y: 0})
                Body.setPosition(body, trajectories[id][s].position)
                Body.setAngularVelocity(body, 0)
                Body.setAngle(body, trajectories[id][s].angle)
                }

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

            if (!_isBrowser && !(typeof opt.do_not_save_img !== 'undefined' &&  opt.do_not_save_img)) {
                demo.render.context.globalAlpha = 0.5
                // if (s == 24 || s == 25 || s == 26) {
                //     demo.render.context.fillStyle = 'white'
                //     console.log(s,'white')
                // } else {
                    demo.render.context.fillStyle = 'white'
                    demo.render.context.fillRect(0,0,demo.width,demo.height)
                    demo.render.context.fillStyle = 'transparent'
                    demo.render.context.fillRect(0,0,demo.width,demo.height)
                    console.log(s,'transparent')
                // }
                // demo.render.context.fillStyle = "rgba(255, 255, 255, 1.0)";
                demo.render.context.fillRect(0,0,demo.width,demo.height)
                Render.world(demo.render)
                let filename = opt.out_folder + '/' + opt.exp_name + '_' + opt.batch_name + '_ex' + opt.ex + '_step' + s +'.png'

                // let filename = 'out'+s+'_'+s+'.png'  // TODO! rename
                PImage.encodePNG(demo.render.canvas, fs.createWriteStream(filename), function(err) {
                    console.log("wrote out the png file to "+filename);
                });

            }

            s++;
            if( s < num_steps ){
                if (_isBrowser) {
                    setTimeout( f, 100 );
                } else {
                    setTimeout( f, 0 );
                }
            }
        }
        // setInterval(function(){
        //   console.log('test');
        // }, 1000);
        f();

        if (config.env == 'tower') {
            console.log('Fraction unstable',fraction_unstable(trajectories,1))
            return [is_stable_trajectory(trajectories) < stability_threshold, is_stable_trajectory(trajectories), fraction_unstable(trajectories,1)]  // true if unstable
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
                }, {
                    option: 'noimg',
                    alias: 'i',
                    type: 'Boolean',
                    description: 'do not save image',
                    required: false
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
        let exp_name = path.basename(path.dirname(experiment_folder))
        let jsons = fs.readdirSync(experiment_folder)

        for (let j=0; j < jsons.length; j++) {
            let jf = jsons[j]
            if (jf.indexOf('batch') !== -1) {
                let loaded_json = jsonfile.readFileSync(experiment_folder + '/' + jf)
                let batch_name = jf.slice(0, -1*'.json'.length)
                let out_folder = experiment_folder + '/../visual/' + batch_name

                let stability_dists = {}

                if (loaded_json.config.env=='tower') {
                    let num_stable = 0
                    let num_unstable = 0
                    for (let b=5; b < 6; b ++) {
                        let options = {out_folder: out_folder, ex: b, exp_name: exp_name, batch_name: batch_name, do_not_save_img: cmd_options.noimg}
                        console.log(batch_name)
                        let is_stable_data = Demo.run(loaded_json, options)
                        let is_stable = is_stable_data[0]
                        let euc_dist_stable = is_stable_data[1]
                        let frac_unstable = is_stable_data[2]
                        console.log('euc dist: ' + euc_dist_stable)
                        stability_dists[batch_name+'_ex'+b] = {is_stable: euc_dist_stable, frac_unstable: frac_unstable};
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
                    jsonfile.writeFileSync(out_folder+'/stability_stats.json', stability_dists=stability_dists)
                    console.log('Wrote to ' + out_folder+'/stability_stats.json')
                } else {
                    let options = {out_folder: out_folder, ex: 0, exp_name: exp_name, batch_name: batch_name, do_not_save_img: cmd_options.noimg}
                    console.log(batch_name)
                    Demo.run(loaded_json, options)
                    console.log('>>>>>>>>>>>>>>>>>>>>>>>>>')
                }
            }
            
        }
        // console.log(jsons)

        // instead of a for loop I should use a call back.

        // but if it is asynchronous may be it will be faster?   




    }
})();


// TODO pass in the example in the batch as well as the experiment folder. You can create a image folder if you want 



