# OpenSCENARIO File Structure

OpenSCENARIO is an ASAM standard for describing dynamic content in driving simulations.
An `.xosc` file is XML-based and follows a strict hierarchy.

## Top-Level Structure

```xml
<OpenScenario>
  <FileHeader />
  <ParameterDeclarations />
  <CatalogLocations />
  <RoadNetwork />
  <Entities />
  <Storyboard />
</OpenScenario>
```

## Key Sections

### FileHeader

Metadata: description, author, date, revMajor/revMinor (version).

### ParameterDeclarations

Named parameters that can be reused across the file.
Useful for parameterizing speed, position, or timing to generate scenario variants.

```xml
<ParameterDeclarations>
  <ParameterDeclaration name="EgoSpeed" parameterType="double" value="10.0" />
</ParameterDeclarations>
```

### RoadNetwork

Points to an OpenDRIVE (`.xodr`) file that defines the road geometry.

### Entities

Defines all actors: the ego vehicle and scenario objects (vehicles, pedestrians, misc objects).

### Storyboard

The main execution logic. Contains:

- **Init** — initial positions and states of all entities
- **Story → Act → ManeuverGroup → Maneuver → Event → Action** — hierarchical trigger-action chain
- **StopTrigger** — condition that ends the scenario

## Trigger → Action Flow

```xml
<Event name="LaneChangeEvent" priority="overwrite">
  <Action name="LaneChangeAction">
    <GlobalAction> ... </GlobalAction>
  </Action>
  <StartTrigger>
    <ConditionGroup>
      <Condition name="SimTimeCondition" ...>
        <ByValueCondition>
          <SimulationTimeCondition value="3.0" rule="greaterThan" />
        </ByValueCondition>
      </Condition>
    </ConditionGroup>
  </StartTrigger>
</Event>
```

## References

- [ASAM OpenSCENARIO Official Page](https://www.asam.net/standards/detail/openscenario/)
- ASAM OpenSCENARIO v1.2 DSC Specification
