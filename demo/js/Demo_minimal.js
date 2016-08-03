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
        var sleep = require('sleep')
        // var PImage = require('pureimage');
        require('./Examples')
        var env = process.argv.slice(2)[0]
        if (env == null)
            throw('Please provide an enviornment, e.g. node Demo.js hockey')
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

    // Create the engine
    Demo.run = function(json_data) {

        //TODO: note that here you should load the demo engine with the json file

        // load the config file here.
        let data = json_data.trajectories
        let config = json_data.config
        console.log(config)

        // let data = json_data

        var demo = {}
        demo.engine = Engine.create()
        demo.runner = Engine.run(demo.engine)
        demo.container = document.getElementById('canvas-container');
        // demo.render = Render.create({element: demo.container, engine: demo.engine})
        // Render.run(demo.render)

        demo.offset = 5;  // world offset
        // demo.cx = 400;
        // demo.cy = 300;

        config.cx = 400;
        config.cy = 300;

        demo.cx = config.cx;
        demo.cy = config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                            max: { x: demo.width, y: demo.height }}

        var world_border = Composite.create({label:'Border'});

        Composite.add(world_border, [
            Bodies.rectangle(demo.cx, -demo.offset, demo.width + 2*demo.offset, 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(demo.cx, demo.height+demo.offset, demo.width + 2*demo.offset, 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(demo.width + demo.offset, demo.cy, 2*demo.offset, demo.height + 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(-demo.offset, demo.cy, 2*demo.offset, demo.height + 2*demo.offset, { isStatic: true, restitution: 1 })
        ]);

        World.add(demo.engine.world, world_border)  // its parent is a circular reference!

        demo.render = Render.create({element: demo.container, engine: demo.engine, 
                                    hasBounds: true, options:{height:demo.height, width:demo.width}})
        Render.run(demo.render)

        Events.on(demo.render, 'afterRender', function(e) {
            var img = demo.render.canvas.toDataURL("image"+e.timestamp+".png")
            var data = img.replace(/^data:image\/\w+;base64,/, "");
            // var buf = new Buffer(data, 'base64');
            // fs.writeFile('image.png', buf);
            // document.write('<img src="'+img+'"/>');  // how do I save this?
            // PImage.encodePNG(img, fs.createWriteStream('out.png'), function(err) {
            // console.log("wrote out the png file to out.png");
            // });
        });

        if (demo.render) {
            var renderOptions = demo.render.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = false;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.background = '#fff';

            if (_isMobile) {
                renderOptions.showDebug = true;
            }
        }

        var mass_colors = {'1':'#C7F464', '15':'#FF6B6B', '30':'#4ECDC4'}// TODO eventually call Example[config.env].mass_colors


        Example[config.env](demo, config)

        // Ok, now let's manually update
        Runner.stop(demo.runner)

        var trajectories = data[0]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length

        console.log(trajectories)

        var i = 0
        function f() {
            console.log( i );
            var entities = Composite.allBodies(demo.engine.world)
                .filter(function(elem) {
                            return elem.label === 'Entity';
                        })
            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
                var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                body.render.fillStyle = mass_colors[trajectories[id][i].mass]//'#4ECDC4'
                if (i < config.num_past) {
                    body.render.strokeStyle = '#FFA500'// orange #551A8B is purple
                } else {
                    body.render.strokeStyle = '#551A8B'// orange #551A8B is purple
                }
                body.render.lineWidth = 5

                // console.log(body)

                Body.setPosition(body, trajectories[id][i].position)
                Body.setVelocity(body, trajectories[id][i].velocity)
                if (trajectories[id][i].mass == 1) {
                    console.log(trajectories[id][i].velocity)
                    console.log(trajectories[id][i].position)
                }
            }

            Runner.tick(demo.runner, demo.engine);
            i++;
            if( i < num_steps ){
                setTimeout( f, 200 );
            }
        }
        f();


    }

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        window.loadFile = function loadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                Demo.run(window.CircularJSON.parse(fr.result))
            }
            fr.readAsText(file)
        }
    }
})();
