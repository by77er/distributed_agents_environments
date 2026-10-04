// Which view a place shows, inside the frame that stays put.

import { Shell } from "./layout/Shell";
import { usePlace } from "./lib/places";
import { Episode } from "./pages/Episode";
import { NewRun } from "./pages/NewRun";
import { GroupView, StepView } from "./pages/Group";
import { Run } from "./pages/Run";
import { Outside, Runs } from "./pages/Runs";
import { Statistics } from "./pages/Statistics";
import { Checkpoint } from "./pages/Checkpoint";
import { Checkpoints } from "./pages/Checkpoints";
import { Evals } from "./pages/Evals";
import { EvalRun } from "./pages/EvalRun";
import { Suite } from "./pages/Suite";
import { SubjectView } from "./pages/Subject";
import { Environment, Environments } from "./pages/Environments";
import { MachineHost, Machines } from "./pages/Machines";

function View() {
  const place = usePlace();
  switch (place.kind) {
    case "run": return <Run name={place.run} />;
    case "step": return <StepView run={place.run} number={place.number} />;
    case "group": return <GroupView run={place.run} number={place.number} />;
    // (an episode's view is its own while one moves between its rollouts; another episode starts afresh)
    case "episode": return <Episode key={place.id} id={place.id} slot={place.slot} />;
    case "outside": return <Outside />;
    case "launch": return <NewRun />;
    case "checkpoints": return <Checkpoints />;
    case "checkpoint": return <Checkpoint id={place.id} />;
    case "evals": return <Evals />;
    case "suite": return <Suite name={place.suite} />;
    case "eval": return <EvalRun run={place.run} />;
    case "subject": return <SubjectView key={`${place.subject}:${place.id}`} kind={place.subject} id={place.id} />;
    case "environments": return <Environments />;
    case "environment": return <Environment key={place.environment} name={place.environment} />;
    case "statistics": return <Statistics />;
    case "machines": return <Machines />;
    case "host": return <MachineHost host={place.host} />;
    default: return <Runs />;
  }
}

export function App() {
  return <Shell><View /></Shell>;
}
