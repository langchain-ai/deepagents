import type { FC } from "react";
import { Streamdown } from "streamdown";

type Props = {
  text: string;
};

const TextPart: FC<Props> = ({ text }) => (
  <div className="markdown-body">
    <Streamdown>{text}</Streamdown>
  </div>
);

export default TextPart;
